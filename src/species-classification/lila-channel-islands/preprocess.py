# script for cropping dataset images to bounding boxes predicted by megadetector v5b

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from google.cloud import storage
from PIL import Image
import io
import json
from tqdm import tqdm

# Initialize Google Cloud Storage client
client = storage.Client()

# Define bucket name and paths - change for different datasets
bucket_name = 'lila_channel_islands_data' #CHANGE THIS: your bucket name
input_json_path = 'channel-islands-camera-traps_mdv5b.0.0_results.json' #CHANGE THIS: path in bucket to where you stored the megadetector results
input_images_folder = 'images/' #CHANGE THIS: path in bucket to folder with original LILA images
output_images_folder = 'cropped_images/' #CHANGE THIS: path in bucket you want to store the cropped images

# Read JSON file from GCS
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(input_json_path)

try:
    json_data = json.loads(blob.download_as_text())
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    exit()

# Function to process a single image


def process_image(entry):
    image_path = entry.get('file', '')
    detections = entry.get('detections', [])

    if image_path.startswith('images/'):
        output_image_path = output_images_folder + \
            image_path.split('images/')[1]
    else:
        output_image_path = output_images_folder + image_path

    # if image is not already cropped
    if not bucket.blob(output_image_path).exists():
        # if its a valid image path
        if bucket.blob(image_path).exists():
            image_blob = bucket.blob(image_path)
            image_data = image_blob.download_as_bytes()

            try:
                image = Image.open(io.BytesIO(image_data))
                img_w, img_h = image.size

                for detection in detections:  # get bounding box
                    bbox = detection.get('bbox', [])
                    conf = detection.get('conf', 0)

                    if conf > .90 and len(bbox) == 4:

                        # crop image
                        xmin = int(bbox[0] * img_w)
                        ymin = int(bbox[1] * img_h)
                        box_w = int(bbox[2] * img_w)
                        box_h = int(bbox[3] * img_h)
                        cropped_image = image.crop(
                            box=[xmin, ymin, xmin + box_w, ymin + box_h])

                        # Upload cropped image to GCS
                        output_blob = bucket.blob(output_image_path)
                        with io.BytesIO() as output_image_data:
                            cropped_image.save(
                                output_image_data, format='JPEG')
                            output_image_data.seek(0)
                            output_blob.upload_from_file(
                                output_image_data, content_type='image/jpeg')
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")


total_images = len(json_data.get('images', []))
with tqdm(total=total_images, desc="Processing Images", unit="image") as pbar:
    def update_progress(*args, **kwargs):
        pbar.update()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_image, entry)
                   for entry in json_data.get('images', [])]
        for future in futures:
            future.add_done_callback(update_progress)

# Close the progress bar after completion
pbar.close()

