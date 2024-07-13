import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from facebox2.utils.image_utils import pil2cv
import tensorflow_io as tfio
import keras

with open('quality_estimator_dataset.pkl', 'rb') as f:
    images, quality_scores = pickle.load(f)

X_train, X_val, y_train, y_val = train_test_split(images, quality_scores, test_size=0.2, random_state=42)

tf.config.run_functions_eagerly(True)


batch_size = 32
learning_rate = 0.001

def load_image(image_path):
    img = tfio.image.decode_webp(tf.io.read_file(image_path))
    img = tfio.experimental.color.rgba_to_rgb(img)
    img = tf.image.resize(img, (112, 112))
    return img

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(lambda x, y: (load_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1024)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.map(lambda x, y: (load_image(x), y))
val_dataset = val_dataset.batch(batch_size)

def create_quality_model(input_shape=(112, 112, 3)):
    model = models.Sequential([
        layers.InputLayer(shape=input_shape),
        layers.Rescaling(1./255),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output layer for the quality score
    ])
    return model


model = create_quality_model()
model.summary()
model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mse', metrics=['mae'])

if True:
    import wandb
    from wandb.integration.keras import WandbMetricsLogger

    wandb.init(project='quality-estimator')
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        keras.callbacks.ModelCheckpoint('quality_estimator.keras', save_best_only=True),
        WandbMetricsLogger()
    ]

    history = model.fit(train_dataset, epochs=200, validation_data=val_dataset, callbacks=callbacks)

    wandb.finish()

    # Plot training history
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
else:
    model.load_weights('quality_estimator.keras')

# visualize predictions using val data
val_images = X_val[:25]
val_scores = y_val[:25]
val_images = [load_image(x) for x in val_images]
val_images = np.array(val_images)
val_scores = np.array(val_scores)
predictions = model.predict(val_images)
predictions = predictions.flatten()

# plot the images and their predicted quality scores in a grid
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
for i, (img, score, pred) in enumerate(zip(val_images, val_scores, predictions)):
    img = img.astype(np.uint8)
    ax = axes[i // 5, i % 5]
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'True: {score:.2f}, Pred: {pred:.2f}')
plt.tight_layout()
plt.show()