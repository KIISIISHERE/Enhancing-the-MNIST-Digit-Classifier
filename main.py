import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Adjust output path to save directly to the visualizations folder
VIS_DIR = "../visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

def main():
    print("⏳ Step 1: Loading MNIST Handwritten Digit Dataset...")
    # 4. Download and Setup MNIST dataset using built-in loader
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel intensity values between 0.0 and 1.0
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape images to include a single channel dimension (28x28x1) for the CNN
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print("🛠️ Step 2: Designing Advanced Neural Network Architecture...")
    # 2 & 3. Combine Data Augmentation, LeakyReLU, and Advanced layers
    model = models.Sequential([
        # Data Augmentation Layers to artificially increase variability
        layers.Input(shape=(28, 28, 1)),
        layers.RandomRotation(0.1),  # Randomly rotate image by 10%
        layers.RandomZoom(0.1),      # Randomly zoom image by 10%

        # First Convolutional Block
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.LeakyReLU(negative_slope=0.1), # Advanced Activation function optimization
        layers.MaxPooling2D((2, 2)),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.LeakyReLU(negative_slope=0.1),
        layers.MaxPooling2D((2, 2)),

        # Flatten & Dense Dropout Layer to avoid overfitting
        layers.Flatten(),
        layers.Dense(128),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax') # 10 Output classes (Digits 0-9)
    ])

    print("⚙️ Step 3: Compiling Model with RMSprop Optimizer...")
    # 3. Optimize Hyperparameters using the RMSprop algorithm
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("🚀 Step 4: Starting Model Training Sequence (3 Epochs)...")
    # Training for 3 epochs keeps execution times super fast while hitting high accuracy
    history = model.fit(
        x_train, y_train, 
        epochs=3, 
        batch_size=64, 
        validation_split=0.1
    )

    print("\n📊 Step 5: Evaluating Model Performance on Test Set...")
    # 5. Evaluate accuracy and loss on unseen test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"\n✅ Test Evaluation Results -> Loss: {test_loss:.4f} | Accuracy: {test_acc*100:.2f}%")

    print("\n📈 Step 6: Generating and Saving Performance Visualizations...")
    # Plot accuracy and loss curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Categorical Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'training_performance.png'))
    plt.close()

    # Visualize predictions compared to actual values
    predictions = model.predict(x_test[:5])
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(10, 3))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(x_test[i].squeeze(), cmap='gray')
        plt.title(f"Pred: {predicted_labels[i]}\nTrue: {y_test[i]}", 
                  color='green' if predicted_labels[i] == y_test[i] else 'red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'digit_predictions_sample.png'))
    plt.close()

    print(f"🎉 Success! Visual evaluation graphs saved cleanly inside: '{VIS_DIR}'")

if __name__ == "__main__":
    main()