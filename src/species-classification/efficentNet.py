from google.cloud import storage
import pandas as pd
import numpy as np
import tensorflow as tf
import io
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.efficientnet import EfficientNetB0

# Initialize Google Cloud Storage client
client = storage.Client()

# Define bucket name and paths
bucket_name = 'lila_channel_islands_data'
labels_csv_path = 'lila_image_urls_and_labels.csv'
image_ids_json_path = 'channel_islands_camera_traps.json'
images_folder = 'cropped_images/'

# Download labels CSV from GCP bucket
labels_blob = client.bucket(bucket_name).blob(labels_csv_path)
labels_csv_data = labels_blob.download_as_text()
labels_df = pd.read_csv(io.StringIO(labels_csv_data))

# filter labels df to channel islands and extract species and preprocessed image ids
labels_df = labels_df[labels_df['image_id'].str.startswith(
    'Channel Islands Camera Traps :')]
labels_df['id'] = labels_df['image_id'].apply(lambda x: x.split(': ')[1])
labels_df = labels_df[['id', 'species']]

# create image id and file path mappings
id_blob = client.get_bucket(bucket_name).blob(image_ids_json_path)
json_data = json.loads(id_blob.download_as_text())
file_to_id_dict = {entry['file_name']: entry['id']
                   for entry in json_data['images']}

# Download images and preprocess them
images = []
labels = []
for filename in file_to_id_dict.keys():
    image_blob = client.bucket(bucket_name).blob(images_folder + filename)
    image_data = image_blob.download_as_bytes()
    # Decode image with 3 channels (RGB)
    image = tf.image.decode_image(image_data, channels=3)
    # Resize image to match EfficientNet input size
    image = tf.image.resize(image, [224, 224])
    # Preprocess image using EfficientNet preprocessing function
    image = preprocess_input(image)
    images.append(image)

    image_id = file_to_id_dict[filename]
    label = labels_df[labels_df['id'] == image_id]['species'].values[0]
    labels.append(label)

NUM_CLASSES = labels.nunique()
# Convert labels to one-hot encoding
y_data = to_categorical(labels, num_classes=NUM_CLASSES)

# Convert image data to numpy array
x_data = np.array(images)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42)

# Load pre-trained EfficientNetB0 model without top classification layers
base_model = EfficientNetB0(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
x = GlobalAveragePooling2D()(base_model.output)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tune the model
model.fit(x_train, y_train, batch_size=32, epochs=10,
          validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print(f'Test accuracy: {accuracy * 100:.2f}%')
