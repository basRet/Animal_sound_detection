import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_io as tfio

def create_model():
    # import data
    x_train, y_train, x_test, y_test = get_animal_audio_data()


    model = tf.keras.models.Sequential([
    # Build your CNN model here
    tf.keras.layers.InputLayer(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3,3), activation='gelu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (4,4), activation='gelu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='gelu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

    epochs = 3  # Try different numbers
    batch_size = 128  # Try different sizes
    optimizer = "adam"  # Try different optimizers
    validation_split = 0.2  # Try different splits

    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    # see metrics

    # Plot Training and Validation Loss
    plt.figure(figsize=(5, 5))
    plt.plot(history.history['loss'], label='Train Loss', color='orange')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    # Print test accuracy
    print("Test Accuracy:", test_accuracy)

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    conf_matrix = confusion_matrix(y_test, predicted_classes)

    sns.heatmap(conf_matrix, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title('Confusion Matrix')
    plt.show()
    return model

def detect_sound(model, audio_filepath):
    model.predict(audio_filepath)

def get_animal_audio_data():
    audio = tfio.audio.audioIOTensor('Animal-Sound-Dataset/cat-Part1/cat_1.wav')

    x_train, y_train, x_test, y_test = 1, 1, 1, 1
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    get_animal_audio_data()
    print("hi")