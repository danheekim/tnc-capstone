import os
import json
import argparse
from pathlib import Path
import boto3
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--coco-file", help="path to coco file")
parser.add_argument("--output-dir", help="local directory to download images to")
args = parser.parse_args()

os.environ['AWS_PROFILE'] = 'animl'
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
sess = boto3.Session()

ENV = 'prod'
SERVING_BUCKET = f'animl-images-serving-{ENV}'

def download_image_file(record, dest_dir, src_bkt=SERVING_BUCKET):
    key = record["serving_bucket_key"]
    relative_dest = record["file_name"]
    full_dest_path = os.path.join(dest_dir, relative_dest)
    Path(full_dest_path).parents[0].mkdir(parents=True, exist_ok=True)
    try:
        boto3.client('s3').download_file(src_bkt, key, full_dest_path)
        return 1  # Successfully downloaded
    except Exception as e:
        print(f"An exception occurred while downloading {key}:")
        print(e)
        return 0  # Failed to download

def download_image_files(img_records, dest_dir):
    print(f"Downloading {len(img_records)} image files to {dest_dir}")
    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers as needed
        results = list(tqdm(executor.map(lambda x: download_image_file(x, dest_dir), img_records), total=len(img_records)))

    successful_downloads = sum(results)
    print(f"Successfully downloaded {successful_downloads} images")

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

if __name__ == "__main__":
    if args.coco_file and args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cct = load_json(args.coco_file)
        download_image_files(cct["images"], args.output_dir)
    else:
        print("Supply a COCO file and output directory")
        print("Run download_images.py --help for usage info")
