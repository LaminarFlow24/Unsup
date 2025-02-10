import os
import io
import shutil
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask_socketio import SocketIO, emit
from PIL import Image
from tensorflow.keras.regularizers import l2

# Import pre-trained helper functions from the separate module.
from pretrained_helpers import (
    pretrained_face_detection,
    pretrained_face_embedding,
    pretrained_object_detection,
    pretrained_image_segmentation,
    pretrained_hand_landmark_detection,
    pretrained_body_landmark_detection,
    apply_pretrained_model,
    PRETRAINED_OUTPUT_TYPE,
    apply_stacked_pretrained_models
)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure secret key

# Initialize SocketIO using threading mode
socketio = SocketIO(app, async_mode='threading')

# Directories for image uploads and model storage
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'model'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 1024  # 1 GB

#############################################
# Model creation functions
#############################################
def create_cnn_model(input_shape, learning_rate, num_conv_layers, conv_filters,
                     kernel_size, stride, padding, dense_layers, dropout_rates, l2_reg, num_classes):
    model = Sequential()
    for i in range(num_conv_layers):
        filters = conv_filters[i]
        if i == 0:
            model.add(Conv2D(filters=filters,
                             kernel_size=(kernel_size, kernel_size),
                             strides=(stride, stride),
                             padding=padding,
                             activation='relu',
                             input_shape=input_shape))
        else:
            model.add(Conv2D(filters=filters,
                             kernel_size=(kernel_size, kernel_size),
                             strides=(stride, stride),
                             padding=padding,
                             activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    for i, neurons in enumerate(dense_layers):
        model.add(Dense(neurons, activation='relu',
                        kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None))
        if i < len(dropout_rates):
            dr = dropout_rates[i]
            if dr > 0:
                model.add(Dropout(dr))
    if num_classes > 2:
        model.add(Dense(num_classes, activation='softmax',
                        kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None))
        loss = 'categorical_crossentropy'
    else:
        model.add(Dense(1, activation='sigmoid',
                        kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None))
        loss = 'binary_crossentropy'
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def create_fc_model(input_dim, learning_rate, dense_layers, dropout_rates, l2_reg, num_classes):
    model = Sequential()
    for i, neurons in enumerate(dense_layers):
        if i == 0:
            model.add(Dense(neurons, activation='relu', input_dim=input_dim,
                            kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None))
        else:
            model.add(Dense(neurons, activation='relu',
                            kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None))
        if i < len(dropout_rates):
            dr = dropout_rates[i]
            if dr > 0:
                model.add(Dropout(dr))
    if num_classes > 2:
        model.add(Dense(num_classes, activation='softmax',
                        kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None))
        loss = 'categorical_crossentropy'
    else:
        model.add(Dense(1, activation='sigmoid',
                        kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None))
        loss = 'binary_crossentropy'
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

#############################################
# Custom generator for vector branch
#############################################
def vector_generator(file_paths, labels, batch_size, preprocessing_fn):
    num_samples = len(file_paths)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_paths = file_paths[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]
            batch_data = []
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                img = img.resize((64,64))
                img_arr = np.array(img).astype(np.float32)/255.0
                if preprocessing_fn is not None:
                    processed = preprocessing_fn(img_arr)
                else:
                    processed = img_arr.flatten()
                if len(processed.shape) > 1:
                    processed = processed.flatten()
                batch_data.append(processed)
            X = np.array(batch_data)
            from tensorflow.keras.utils import to_categorical
            y = to_categorical(batch_labels, num_classes=len(set(labels))) if len(set(labels)) > 2 else np.array(batch_labels)
            yield X, y

#############################################
# Routes
#############################################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_classes', methods=['GET', 'POST'])
def set_classes():
    if request.method == 'POST':
        class_names = request.form.get('class_names')
        if class_names:
            session['classes'] = [name.strip() for name in class_names.split(',') if name.strip()]
            flash("Classes set successfully!")
            return redirect(url_for('upload_images'))
        else:
            flash("Please enter at least one class name.")
    return render_template('set_classes.html')

@app.route('/upload_images')
def upload_images():
    class_names = session.get('classes', [])
    return render_template('upload_images.html', class_names=class_names)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image file part", 400
    image_file = request.files['image']
    if image_file.filename == '':
        return "No file selected", 400
    img_class = request.form.get('img_class')
    if not img_class:
        return "No class specified", 400
    target_folder = os.path.join(app.config['UPLOAD_FOLDER'], img_class)
    os.makedirs(target_folder, exist_ok=True)
    filename = secure_filename(image_file.filename)
    image_file.save(os.path.join(target_folder, filename))
    return "File uploaded successfully", 200

@app.route('/start_training', methods=['GET', 'POST'])
def start_training():
    if request.method == 'POST':
        # Retrieve basic hyperparameters
        learning_rate = float(request.form.get('learning_rate', 0.001))
        batch_size = int(request.form.get('batch_size', 32))
        epochs = int(request.form.get('epochs', 10))
        kernel_size = int(request.form.get('kernel_size', 3))
        stride = int(request.form.get('stride', 1))
        padding = request.form.get('padding', 'same')
        num_conv_layers = int(request.form.get('num_conv_layers', 1))
        conv_filters_str = request.form.get('conv_filters', '32')
        conv_filters = [int(x.strip()) for x in conv_filters_str.split(',') if x.strip()]
        if len(conv_filters) != num_conv_layers:
            flash("Number of convolution filters does not match the number of conv layers.")
            return redirect(url_for('start_training'))
        
        # Retrieve new hyperparameters for the dense layers
        dense_layers_str = request.form.get('dense_layers', '')
        if dense_layers_str:
            dense_layers = [int(x.strip()) for x in dense_layers_str.split(',') if x.strip()]
        else:
            dense_layers = []
        
        dropout_rates_str = request.form.get('dropout_rates', '')
        if dropout_rates_str:
            dropout_rates = [float(x.strip()) for x in dropout_rates_str.split(',') if x.strip()]
        else:
            dropout_rates = []
        
        l2_reg = float(request.form.get('l2_reg', 0.0))
        
        # Retrieve custom classes from session; default to two classes if not set.
        classes = session.get('classes', [])
        if len(classes) < 2:
            classes = ["class1", "class2"]
        num_classes = len(classes)
        class_mode = 'categorical' if num_classes > 2 else 'binary'
        
        # Retrieve stacked pretrained model options (comma separated)
        stacked_pretrained_options = request.form.get('stacked_pretrained_options', 'None')
        session['stacked_pretrained_options'] = stacked_pretrained_options
        if stacked_pretrained_options.strip().lower() == "none":
            preprocessing_fn = None
        else:
            options = [o.strip() for o in stacked_pretrained_options.split(',') if o.strip() != ""]
            def preprocessing_fn(x):
                return apply_stacked_pretrained_models(x, options)
        
        # Determine branch based on output type: vector if any option is vector-producing.
        vector_models = ["Face Embedding", "Hand Landmark Detection", "Body Landmark Detection"]
        branch = "image"
        if stacked_pretrained_options.strip().lower() != "none":
            for opt in options:
                if opt in vector_models:
                    branch = "vector"
                    break
        session['branch'] = branch
        
        if branch == "image":
            target_size = (64,64)
            datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2,
                preprocessing_function=preprocessing_fn
            )
            train_generator = datagen.flow_from_directory(
                app.config['UPLOAD_FOLDER'],
                target_size=target_size,
                batch_size=batch_size,
                class_mode=class_mode,
                subset='training',
                classes=classes
            )
            validation_generator = datagen.flow_from_directory(
                app.config['UPLOAD_FOLDER'],
                target_size=target_size,
                batch_size=batch_size,
                class_mode=class_mode,
                subset='validation',
                classes=classes
            )
            input_shape = (64,64,3)
            model = create_cnn_model(input_shape, learning_rate, num_conv_layers, conv_filters,
                                       kernel_size, stride, padding, dense_layers, dropout_rates, l2_reg, num_classes)
            steps_per_epoch = train_generator.samples // batch_size
            if steps_per_epoch < 1:
                steps_per_epoch = 1
            validation_steps = validation_generator.samples // batch_size
            if validation_steps < 1:
                validation_steps = 1
            train_data = train_generator
            val_data = validation_generator
        else:
            dummy = np.zeros((64,64,3), dtype=np.float32)/255.0
            if preprocessing_fn is not None:
                dummy_processed = preprocessing_fn(dummy)
            else:
                dummy_processed = dummy.flatten()
            if len(dummy_processed.shape) > 1:
                dummy_processed = dummy_processed.flatten()
            feature_dim = dummy_processed.shape[0]
            input_shape = (feature_dim,)
            model = create_fc_model(input_dim=feature_dim, learning_rate=learning_rate,
                                    dense_layers=dense_layers, dropout_rates=dropout_rates, l2_reg=l2_reg, num_classes=num_classes)
            train_file_paths = []
            train_labels = []
            val_file_paths = []
            val_labels = []
            for i, cls in enumerate(classes):
                cls_folder = os.path.join(app.config['UPLOAD_FOLDER'], cls)
                if os.path.exists(cls_folder):
                    files = sorted(os.listdir(cls_folder))
                    n = len(files)
                    train_count = int(n * 0.8)
                    for file in files[:train_count]:
                        train_file_paths.append(os.path.join(cls_folder, file))
                        train_labels.append(i)
                    for file in files[train_count:]:
                        val_file_paths.append(os.path.join(cls_folder, file))
                        val_labels.append(i)
            steps_per_epoch = len(train_file_paths) // batch_size
            if steps_per_epoch < 1:
                steps_per_epoch = 1
            validation_steps = len(val_file_paths) // batch_size
            if validation_steps < 1:
                validation_steps = 1
            train_data = vector_generator(train_file_paths, train_labels, batch_size, preprocessing_fn)
            val_data = vector_generator(val_file_paths, val_labels, batch_size, preprocessing_fn)
        
        class SocketIOCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    logs = {}
                print(f"Epoch {epoch+1} complete. Logs: {logs}")
                socketio.emit('epoch_update', {
                    'epoch': epoch + 1,
                    'train_accuracy': logs.get('accuracy', 0),
                    'val_accuracy': logs.get('val_accuracy', 0)
                })
        
        def background_train():
            with app.app_context():
                print("Background training started")
                history = model.fit(
                    train_data,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=val_data,
                    validation_steps=validation_steps,
                    callbacks=[SocketIOCallback()]
                )
                print("Background training finished")
                model_path = os.path.join(MODEL_FOLDER, 'cnn_model.h5')
                model.save(model_path)
                if os.path.exists(app.config['UPLOAD_FOLDER']):
                    shutil.rmtree(app.config['UPLOAD_FOLDER'])
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                socketio.emit('training_complete', {'message': 'Training complete!'})
        
        socketio.emit('training_started', {'message': 'Training started...'})
        socketio.start_background_task(background_train)
        return render_template('training_progress.html')
    return render_template('start_training.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model_path = os.path.join(MODEL_FOLDER, 'cnn_model.h5')
    if not os.path.exists(model_path):
        flash("Model not found. Please train a model first!")
        return redirect(url_for('index'))
    model = tf.keras.models.load_model(model_path)
    branch = session.get('branch', 'image')
    stacked_pretrained_options = session.get('stacked_pretrained_options', 'None')
    classes = session.get('classes', ["class1", "class2"])
    if len(classes) < 2:
        classes = ["class1", "class2"]
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            target_size = (64, 64)
            image = Image.open(filepath).convert('RGB')
            image = image.resize(target_size)
            img_arr = np.array(image).astype(np.float32)/255.0
            # Apply the same stacked pre-trained models used during training
            if stacked_pretrained_options.strip().lower() != "none":
                options = [o.strip() for o in stacked_pretrained_options.split(',') if o.strip() != ""]
                img_arr = apply_stacked_pretrained_models(img_arr, options)
            if branch == "vector":
                processed = img_arr.flatten() if len(img_arr.shape) > 1 else img_arr
                input_data = np.expand_dims(processed, axis=0)
            else:
                input_data = np.expand_dims(img_arr, axis=0)
            prediction = model.predict(input_data)
            if len(classes) > 2:
                predicted_index = np.argmax(prediction, axis=1)[0]
                predicted_class = classes[predicted_index]
            else:
                predicted_class = classes[1] if prediction[0][0] > 0.5 else classes[0]
            flash(f"Predicted class: {predicted_class}")
            return redirect(url_for('predict'))
    return render_template('predict.html')

if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False)
