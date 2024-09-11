import cv2
import os
import numpy as np
from skimage.io import imread
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

#MAX_MODEL_SIZE = 252234624 

def load_images(destination, image_size=(224, 224)):
    images = []
    labels = []
    for label in os.listdir(destination):  #To differentiate classes
        label_path = os.path.join(destination, label)
        
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                img = cv2.resize(img, image_size)  
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)


def normalize_images(images):
    return images.astype('float32') / 255.0

def encode_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)  
    one_label = to_categorical(encoded_labels) 
    return one_label, label_encoder

def save_data(X_train, X_val, X_test, y_train, y_val, y_test):
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)


def load_data():
    X_train = np.load('X_train.npy')
    X_val = np.load('X_val.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_images():
    image_size = (224, 224)
    dataset_path = r'C:\University\Personal\Project\covid19-detection\dataset'
    images, labels = load_images(dataset_path, image_size)
    images = normalize_images(images)
    one_label, label_encoder = encode_labels(labels)

    X_train, X_test, y_train, y_test = train_test_split(images, one_label, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    
    save_data(X_train, X_val, X_test, y_train, y_val, y_test)

    print("Preprocessing done.")
    print("Shape of samples:", X_train.shape)

if __name__ == "__main__":
    preprocess_images()
    
