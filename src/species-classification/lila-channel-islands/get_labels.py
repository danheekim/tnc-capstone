# Script for getting labels of cropped images

from google.cloud import storage
import pandas as pd
import io
import json
from tqdm import tqdm


# Initialize Google Cloud Storage client
client = storage.Client()

# Define bucket name and paths
bucket_name = 'lila_channel_islands_data' # CHANGE THIS: your bucket name where the original LILA images are stored
labels_csv_path = 'lila_channel_islands_true_labels.csv' # CHANGE THIS: path in bucket to LILA Channel Islands true labels csv 
image_ids_json_path = 'channel_islands_camera_traps.json' # CHANGE THIS: path in bucket to LILA Channel Islands COCO json 
images_folder = 'cropped_images/' # CHANGE THIS: folder in your bucket you want to store the cropped images

# Download labels CSV from GCP bucket
labels_blob = client.bucket(bucket_name).blob(labels_csv_path)
labels_csv_data = labels_blob.download_as_text()
labels_df = pd.read_csv(io.StringIO(labels_csv_data))

# create image id and file path mappings
id_blob = client.get_bucket(bucket_name).blob(image_ids_json_path)
json_data = json.loads(id_blob.download_as_text())
file_to_id_dict = {entry['file_name']: entry['id']
                   for entry in json_data['images']}

# save filename and corresponding label
df = []
for filename in tqdm(file_to_id_dict.keys(), desc="Getting Labels", unit="image file"):
    # if image is in cropped_images folder
    if client.bucket(bucket_name).blob(images_folder + filename).exists():
        image_id = file_to_id_dict[filename]
        label = labels_df[labels_df['image_id'] ==
                          image_id]['original_label'].values[0]
        df.append([filename, label])

# Upload full df to GCS
df = pd.DataFrame(df, columns=['image_file', 'label'])
csv_data = df.to_csv(index=False)
path_name = 'lila_channel_islands_images_and_labels_all.csv' # CHANGE THIS: desired output csv name
df_blob = client.bucket(bucket_name).blob(path_name)
df_blob.upload_from_string(csv_data, content_type='text/csv')
