# train_model.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.mixed_precision import set_global_policy, Policy  # Import mixed precision
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set parameters
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = 'data'  # Update to your dataset path

# Enable mixed precision
set_global_policy(Policy('mixed_float16'))

# Prepare data generators
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

# Function to create datasets
def create_tf_dataset(generator, subset):
    return tf.data.Dataset.from_generator(
        lambda: generator.flow_from_directory(
            DATA_DIR,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset=subset,
            shuffle=True
        ),
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_WIDTH, IMG_HEIGHT, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)  # Caching and prefetching for speed

# Create training and validation datasets
train_data = create_tf_dataset(datagen, 'training')
val_data = create_tf_dataset(datagen, 'validation')

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
x = GlobalAveragePooling2D()(base_model.output)
predictions = Dense(1, activation='sigmoid', dtype='float32')(x)  # Ensure output is float32

# Create model
model = Model(inputs=base_model.input, outputs=predictions)

# # Freeze base model layers
# base_model.trainable = False  # Freeze the entire base model in one line

# Compile model with mixed precision
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('forgery_detection_model.keras', save_best_only=True)
]

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save the model
model.save('forgery_detection_model.keras')

# Plot training & validation accuracy & loss values
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_training_history(history)
