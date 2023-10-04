# script for training efficientNet

from google.cloud import storage
import pandas as pd
import numpy as np
import tensorflow as tf
import io
import json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import pickle
from sklearn.preprocessing import LabelEncoder


############## GCP Stuff #######################################################

# Initialize Google Cloud Storage client
client = storage.Client()

# Define bucket name and paths
bucket_name = 'lila_channel_islands_data'
data_path = 'lila_channel_islands_images_and_labels_temp.csv'
images_folder = 'cropped_images/'

# Download data CSV from GCP bucket
df_blob = client.bucket(bucket_name).blob(data_path)
data = df_blob.download_as_text()
df = pd.read_csv(io.StringIO(data))
num_classes = df['label'].nunique()

############## Data Preprocessing ##############################################

# Function to load and preprocess an image


def load_and_preprocess_image(image_path, label):
    image_blob = client.bucket(bucket_name).blob(
        images_folder + image_path.numpy().decode('utf-8'))
    if image_blob.exists():
        image_data = image_blob.download_as_bytes()
        image = tf.image.decode_image(image_data, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image, label
    else:
        return None


# Create a tf.data.Dataset from labels CSV
labels_dataset = tf.data.Dataset.from_tensor_slices(
    (df['image_file'].values, df['label'].values)
)
labels_dataset = labels_dataset.map(
    lambda x, y: tf.py_function(load_and_preprocess_image, [
                                x, y], [tf.float32, tf.string]),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Filter out None values (images that failed to download)
labels_dataset = labels_dataset.filter(lambda x, y: x is not None)

# One-hot encode the labels


def one_hot_encode(image, label):
    label = tf.one_hot(label, num_classes)
    return image, label


labels_dataset = labels_dataset.map(
    lambda x, y: tf.py_function(one_hot_encode, [
                                x, y], [tf.float32, tf.float32]),
    num_parallel_calls=tf.data.AUTOTUNE
)


# Batch the dataset and split into train test
total_samples = len(df)
labels_dataset = labels_dataset.shuffle(buffer_size=total_samples)

# Define the ratio for splitting the dataset
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2
train_size = int(total_samples * train_ratio)
val_size = int(val_ratio * total_samples)
test_size = int(test_ratio * total_samples)

# Split the dataset into training and testing sets
train_dataset = labels_dataset.take(train_size)
val_dataset = labels_dataset.skip(train_size).take(val_size)
test_dataset = labels_dataset.skip(train_size + val_size)

# Prefetch the dataset for better performance
batch_size = 16
train_dataset = train_dataset.batch(
    batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(
    batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(
    buffer_size=tf.data.AUTOTUNE)

############## Create Model ####################################################

# Load pre-trained EfficientNetB0 model without top classification layers
base_model = EfficientNetB0(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
x = GlobalAveragePooling2D()(base_model.output)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze some layers of the pre-trained model
for layer in base_model.layers[:-20]:  # Freeze all layers except the last 20
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])

############## training ########################################################

epochs = 10
history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

############## save model ######################################################

# Upload model to gcp bucket
pickle_bytes = pickle.dumps(model)
destination_blob_name = 'lila_channel_islands_efficentNet_test.pkl'
model_blob = client.bucket(bucket_name).blob(destination_blob_name)
