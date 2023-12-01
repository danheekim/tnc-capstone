# Anomaly Detection

This module intends to explore anomaly detection methods using autoencoders for invasive
species detection. The data set used in this module is the Santa Cruz Islands camera trap
images downloaded from Animl. The 3 main methods we explore in this module are:

1. Anomaly detection using a traditional deep neural network autoencoder
2. Anomaly detection using deep convolutional autoencoders
3. Visualizing species representations using deep convolutional autoencoders and t-SNE

## Table of Contents
- [Anomaly Detection](#anomaly-detection)
  - [Table of Contents](#table-of-contents)
  - [Module Organization](#module-organization)
  - [Data Download](#data-download)
  - [Preprocessing](#preprocessing)
  - [Anomaly Detection with Traditional Autoencoders](#anomaly-detection-with-traditional-autoencoders)
  - [Anomaly Detection with Convolutional Autoencoders](#anomaly-detection-with-convolutional-autoencoders)
  - [Convolutional Auotencoders, t-SNE & Visualization](#species-repersentation-visualization)


## Module Organization
```
tnc-capstone/
├── src/
│   ├── anomaly-detection/
│   │   ├── README.md
│   │   ├── sci_eda/
│   │   │   ├── sci_eda_after.png
│   │   │   └── sci_eda_before.png
│   │   ├── traditional_anomaly_detection_tsne_and_accuracies/
│   │   │   ├── invasive_bat_acc.png
│   │   │   ├── invasive_bat_tsne.png
│   │   │   └── ...
│   │   ├── cae_reconstructions/
│   │   │   ├── bat_recon.png
│   │   │   ├── fox_recon.png
│   │   │   └── ...
│   │   ├── cae_anomaly_detection_tsne_and_accuracies/
│   │   │   ├── bat_acc_new.png
│   │   │   ├── bat_tsne_new.png
│   │   │   └── ...
│   │   ├── cae_for_visualization/
│   │   │   ├── recon_tsne_p5.png
│   │   │   ├── recon_tsne_p10.png
│   │   │   └── ...
│   │   ├── models/
│   │   │   ├── cae_for_visualization.h5
│   │   │   └── cae_sci_native.h5
│   │   ├── crop_images.py
│   │   ├── download_images_mod.py
│   │   ├── anomaly_detection_basic.ipynb
│   │   ├── CAE_anomaly_detection.ipynb
│   │   └── cae_for_visualizatoin.ipynb
          
```

## Data Download
To download the Santa Cruz Islands (SCI) data from Animl, you must have an Animl
account. To download the data, follow the instructions described below. 

**Step 1:**  
First, make sure you are in the `anomaly-detection` directory. Then run the 
following in your terminal:
1. `pip install aws configure`
2. `pip install awscli`
3. `aws configure --profile animl`  
    
The third command above will prompt you for the following:
* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACESS_KEY`
* `DEFAULT_REGION_NAME` = `us-west-2`
* `DEFAULT_OUTPUT_FORMAT` = `text`
    
The `AWS_ACESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` were provided to us by Nathaniel
Rindlaub (nathaniel.rindlaub@tnc.org) - contact him for details.
  
**Step 2:**  
Once you have configured the AWS environment by following the steps above, follow
the instructions [here](https://github.com/tnc-ca-geo/animl-analytics). 
  
This will clone the `animl-analytics` directory into your `anomaly-detection` directory.

**Step 3:**   
To download the SCI biodiversity dataset images run the following in your terminal:
1. `cd animl-analytics`
2. `python utils/download_images.py --coco-file ../SCI_coco.json --output-dir ../images`  
    
Here, `../SCI_coco.json` is the path to the COCO json file for the SCI dataset, 
downloaded from Animl and `../images` is the path to the directory we would like
to store the images in. You can replace these with your own values.   
  
Note, the COCO json file is only for images that are human reviewed and aren't labeled `empty`, 
`person`, `vehicle`,or `animal`.  We chose these values for simplicity/data cleanliness. 
    
If it takes too long to download the images/the session crashes in the middle, 
you can run the following script instead :  
    
`python utils/download_images_mod.py --coco-file ../SCI_coco.json --output-dir ../images`
    
The script `download_images_mod.py` is a modified version of `download_images.py` that
we created to download multiple images in parallel and skip already downloaded
images. This should be faster than the original script. We have provided the 
`download_images_mod.py` file for your use. Make sure you move it into the 
`animl-analytics\utils` directory before running the above command.
    
After following the above steps, you should have a local folder containing the SCI
camera trap images.  
  
## Preprocessing
The COCO json file for the SCI data contains three fields:
1. `images`: A list where each item is a dictionary for an image with the following fields...
      * `id`: image id 
      * `file_name`: image file path
2. `annotations`: A list where each item is a dictionary for an image with the following fields...
      * `image_id`: image id same as `id` in `images`
      * `category_id`: category id of the image label, matches `id` in `categories`
      * `bbox`: coordinates for the bounding box containing the animal
3. `categories`: A dictionary with the following fields...
      * `id`: label id (ex: 1)
      * `name`: label name (ex: "scrub jay")

Thus using the COCO json, we can create a dataset that contains filenames of the
cropped images (according to `bbox`) and the corresponding species labels. This
intermediary file will make it easier for us to pull in images from a local folder
(containing the cropped images) into Google Colab (where we will be doing our 
exploration).   

The script for creating the dataset above is called `crop_images.py` and
can be run using the following command:    
  
`python crop_images.py`    
  
Running the above script should create a new directory for the cropped images
where the file name corresponds to the `file_name` attribute of the image. You
should also see a csv containing the image file names and the species labels in 
the `anomaly-detection` directory. 

## Anomaly Detection with Traditional Autoencoders

## Anomaly Detection with Convolutional Autoencoders

## Species Repersentation Visualization