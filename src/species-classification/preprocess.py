# script for cropping dataset images to bounding boxes predicted by megadetector v5b

from google.cloud import storage
from PIL import Image
import io
import json

# Initialize Google Cloud Storage client
client = storage.Client()

# Define bucket name and paths - change for different datasets
bucket_name = 'lila_channel_islands_data'
input_json_path = 'channel-islands-camera-traps_mdv5b.0.0_results.json'
input_images_folder = 'images/'
output_images_folder = 'cropped_images/'

# Read JSON file from GCS
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(input_json_path)

try:
    json_data = json.loads(blob.download_as_text())
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    exit()

# Process each entry in the JSON file
for entry in json_data.get('images', []):
    image_path = entry.get('file', '')
    detections = entry.get('detections', [])

    # Check if the image file exists in the bucket
    if bucket.blob(image_path).exists():
        image_blob = bucket.blob(image_path)
        image_data = image_blob.download_as_bytes()
        image = Image.open(io.BytesIO(image_data))
        img_w, img_h = image.size

        # Process each detection in the 'detections' list
        for detection in detections:
            bbox = detection.get('bbox', [])
            conf = detection.get('conf', 0)
            if conf > .90:
                # Ensure bbox contains valid coordinates
                if len(bbox) == 4:
                    # Crop and upload the image
                    xmin = int(bbox[0] * img_w)
                    ymin = int(bbox[1] * img_h)
                    box_w = int(bbox[2] * img_w)
                    box_h = int(bbox[3] * img_h)
                    cropped_image = image.crop(
                        box=[xmin, ymin, xmin + box_w, ymin + box_h])

                    # Upload cropped image to GCS
                    output_image_path = output_images_folder + \
                        image_path.split('images/')[1]
                    output_blob = bucket.blob(output_image_path)
                    with io.BytesIO() as output_image_data:
                        # Save as JPG format
                        cropped_image.save(output_image_data, format='JPEG')
                        output_image_data.seek(0)
                        output_blob.upload_from_file(
                            output_image_data, content_type='image/jpeg')

                    print(f"Cropped image uploaded: {output_image_path}")
                else:
                    print(f"Invalid bbox coordinates for image: {image_path}")
            else:
                print(f'BBox confidence <= .90: {image_path}')
    else:
        print(f"Image file not found in the bucket: {image_path}")
