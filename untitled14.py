import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow.keras as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,MaxPool2D
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import glob
import seaborn as sns
import json
s = 150
dataset = []
label = []

# Directory paths
DFU = 'DFU'
WOUND = 'Wound Images'

# Function to decode and append images
def decode_and_append_images(directory, class_label):
    images = glob.glob(os.path.join(directory, '*'))
    for image in images:
        img = cv2.imread(image)
        if img is not None:  # Ensure image was successfully read
            img = cv2.resize(img, (s, s))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0  # Normalize to [0, 1]
            img = img.astype(np.float32)  # Cast to float32
            dataset.append(img)
            label.append(class_label)

# Decode images for DFU
decode_and_append_images(DFU, 0)

# Decode images for Wound
decode_and_append_images(WOUND, 1)

dataset = np.array(dataset)
label = np.array(label)

print("Dataset shape:", dataset.shape)
print("Label shape:", label.shape)

# Shuffle the data
dataset, label = shuffle(dataset, label, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.25, stratify=label, random_state=43)

# One-hot encode the labels
y_train_one_hot = to_categorical(y_train, num_classes=2)
y_test_one_hot = to_categorical(y_test, num_classes=2)

# Build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(s, s, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False
)

datagen.fit(X_train)

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train_one_hot, batch_size=32),
    epochs=30,
    validation_data=(X_test, y_test_one_hot),
    callbacks=[early_stop]
)



# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_one_hot)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Get predicted probabilities
y_pred_prob = model.predict(X_test)

# Convert predicted probabilities to class labels
y_pred = np.argmax(y_pred_prob, axis=1)

# Use the class labels for evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(y_pred[:10])
print(y_test[:10])

#confusion_matrix
cm = confusion_matrix(y_pred , y_test)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='.0f')
plt.xlabel("Predicted Digits")
plt.ylabel("True Digits")
plt.show()

# Plotting Bar chart
classifiers= [ 'CNN']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
bar_width = 0.2
index = np.arange(len(classifiers))

plt.bar(index, accuracy, bar_width, label='Accuracy')
plt.bar(index + bar_width,precision , bar_width, label='Precision')
plt.bar(index + 2 * bar_width, recall, bar_width, label='Recall')
plt.bar(index + 3 * bar_width, f1, bar_width, label='F1 Score')

plt.xlabel('Classifiers')
plt.ylabel('Metric Values')
plt.title('Metrics for Different Classifiers')
plt.xticks(index +  bar_width / 2, classifiers)
plt.legend()
plt.show()

# Save the model as .h5 file
model.save('your_model_file.h5')
print("Model saved as 'your_model_file.h5'")






#gui predict
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")

        # GUI components
        self.image_label = tk.Label(root)
        self.result_label = tk.Label(root, text="Result: ")

        # Open File button
        self.open_file_button = tk.Button(root, text="Open Image", command=self.open_and_process_image)
        self.open_file_button.pack()

    def open_and_process_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.process_image(file_path)

    def process_image(self, image_path):
        # Load and preprocess the image for model input
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        # Make predictions using the loaded model
        predictions = model.predict(img_array)

        # Get the predicted class index
        predicted_class_index = tf.argmax(predictions, axis=1).numpy().item()  # Extract the scalar value
        print(f"Predicted class index: {predicted_class_index}")

        # Optionally, you can also get the class labels if you have them
        class_labels = ['DFU', 'Wound']  # Replace with your actual class labels
        predicted_class_label = class_labels[predicted_class_index]
        print(f"Predicted class label: {predicted_class_label}")

        # Get class probabilities
        class_probabilities = predictions[0]
        for i, prob in enumerate(class_probabilities):
            print(f"Probability for class '{class_labels[i]}': {prob:.4f}")

        # Display the image and prediction result
        self.display_image_and_result(img, predicted_class_label)

    def display_image_and_result(self, img, result):
        # Update the image label
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img
        self.image_label.pack()

        # Update the result label
        self.result_label.config(text=f"Result: {result}")
        self.result_label.pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
    

