import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Set page title
st.title("MNIST Digit Classification with CNN")

# Load and preprocess MNIST data
@st.cache_data  # Cache data to avoid reloading
def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Normalize and reshape
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()

# Build CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# Train the model (with progress bar)
if st.button("Train Model"):
    with st.spinner("Training in progress..."):
        history = model.fit(
            X_train, y_train, 
            epochs=5, 
            batch_size=64, 
            validation_split=0.2,
            verbose=0
        )
    st.success("Training completed!")
    
    # Plot training history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Accuracy')
    ax[0].legend()
    
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title('Loss')
    ax[1].legend()
    
    st.pyplot(fig)

# Evaluate model
if st.button("Evaluate Model"):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.write(f"Test Accuracy: **{accuracy*100:.2f}%**")
    
    # Display 5 random test images with predictions
    st.subheader("Sample Predictions")
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    for i, idx in enumerate(sample_indices):
        img = X_test[idx].reshape(28, 28)
        true_label = np.argmax(y_test[idx])
        pred_label = np.argmax(model.predict(img.reshape(1, 28, 28, 1)))
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}")
        axes[i].axis('off')
    
    st.pyplot(fig)