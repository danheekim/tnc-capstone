import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import pandas as pd
from tqdm import tqdm

def process_images_batch(images_batch):
    processed_images = []

    for image_info in images_batch:
        image_id = image_info['id']
        image_file_name = image_info['file_name']

        # retrieve the annotation
        annot = next((annotation for annotation in annotations if annotation['image_id'] == image_id), None)

        if not annot:
            tqdm.write(f"Image:{image_id} missing annotation, skipping it.")
            continue

        bbox = annot['bbox']
        # check if the bounding box is valid
        if len(bbox) != 4 or any(not isinstance(coord, (int, float)) for coord in bbox):
            tqdm.write(f"Image:{image_id} has an invalid bounding box, skipping it.")
            continue

        cat_id = annot['category_id']
        species = categories_species.get(cat_id)
        
        if species is None:
            tqdm.write(f"Image:{image_id} has an unknown category, skipping it.")
            continue

        # crop and save image
        input_path = os.path.join(images_folder_path, image_file_name)
        output_path = os.path.join(output_dir_path, os.path.basename(input_path))

        # check if the image file exists
        if not os.path.isfile(input_path):
            tqdm.write(f"Image:{image_id} not found, skipping it.")
            continue

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with Image.open(input_path) as img:
                img_width, img_height = img.size
                # check if the bounding box is within the image dimensions
                if bbox[0] + bbox[2] > img_width or bbox[1] + bbox[3] > img_height:
                    tqdm.write(f"Image:{image_id} has an invalid bounding box size, skipping it.")
                    continue
                cropped_img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                
                # check if the cropped image is empty
                if cropped_img.size[0] == 0 or cropped_img.size[1] == 0:
                    tqdm.write(f"Image:{image_id} has an empty bounding box, skipping it.")
                    continue
                
                cropped_img.save(output_path)
        except Exception as e:
            tqdm.write(f"An error occurred while processing Image:{image_id}: {e}")
            continue

        processed_images.append([output_path, species])

    return processed_images

images_folder_path = 'images/'
output_dir_path = 'cropped_images/'
coco_json_path = 'SCI_coco.json'

# read in COCO json
with open(coco_json_path, "r") as json_file:
    data = json.load(json_file)

images = data['images']
annotations = data['annotations']
categories = data['categories']

# create category to species dict
categories_species = {cat['id']: cat['name'] for cat in categories}

# Batch size for parallel processing
batch_size = 50

# Split images into batches
image_batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]

# Use ThreadPoolExecutor for parallel processing with tqdm progress bar
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_images_batch, batch) for batch in image_batches]

    processed_batches = []
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Batches"):
        processed_batches.extend(future.result())

# Remove None values (skipped images)
processed_images = [item for item in processed_batches if item is not None]

# Save processed_images as a csv
df = pd.DataFrame(processed_images, columns=['image_path', 'species_label'])
df.to_csv('SCI_biodiversity_images_and_labels.csv', index=False)
