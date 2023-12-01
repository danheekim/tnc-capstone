# Anomaly Detection
In this directory, we focus on exploring various anomaly detection methods on the Santa
Cruz Islands (SCI) dataset in Animl. 

## Directory Structure
------------
      └── anomaly-detection/
            ├── README.md
            ├── crop_images.py
            ├── download_images_mod.py
            ├── anomaly_detection_basic.ipynb
            ├── CAE_anomaly_detection.ipynb
            ├── cae_for_visualization.ipynb
            ├── cae_anomaly_detection_tsne_and_accuracies/...
            ├── cae_for_visualization/...
            ├── cae_reconstructions/...
            ├── traditional_anomaly_detection_tsne_and_accuracies/...
            ├── sci_eda/...
            └── models/
                ├── cae_for_visualization.h5
                └──  cae_sci_native.h5  

      
          
            
--------
Description of directory files is given below:
1. `crop_images.py` : python script for cropping images to their COCO json bounding
boxes and matching them to their labels
2. `download_images_mod.py` : modified version of `download_images.py` from 
`animl-analytics` repo that uses parallel processing for faster download
3. `anomaly_detection_basic.ipynb` : notebook for running anomaly detection on the
Santa Cruz Islands data with traditional autoencoder structure and no image feature extraction
4. `CAE_anomaly_detection.ipynb` : notebook for running anomaly detection on the Santa
Cruz Islands data with convolutional autoencoders (preferred)
5. `CAE_for_visualization.ipynb`: notebook for training an convolutional autoencoder on high-level animal
labels and using t-sne to visualize the reconstructions
  
Description of directory folders is given below:
1. `cae_anomaly_detection_tsne_and_accuracies/`: contains images of invasive species t-sne
reconstructions and prediction accuracies from `CAE_anomaly_detection.ipynb`
2. `cae_for_visualization/`: contains images of t-sne reconstructions at different perplexities
for all high level animal labels from `cae_for_visualization.ipynb`
3. `cae_reconstructions/`: contains reconstruction images from the trained model in `CAE_anomaly_detection.ipynb`
4. `traditional_anomaly_detection_tsne_and_accuracies/`: contains images of the invasive 
species t-sne reconstructions and prediction accuracies from `anomaly_detection_basic.ipynb`
5. `sci_eda/`: contains santa cruz island EDA plots from `CAE_anomaly_detection.ipynb`
6. `models/`: contains `cae_sci_native.h5` which is the model from `CAE_anomaly_detection.ipynb`
and `cae_for_visualization.h5` which is the model from `cae_for_visualizatoin.ipynb`

## How to download the SCI dataset from Animl
** NOTE: You must have an Animl account to do download the data! **

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

Once you have configured the AWS environment by following the steps above, follow
the instructions here: https://github.com/tnc-ca-geo/animl-analytics. This will
clone the `animl-analytics` directory into your `anomaly-detection` directory.
Note, we did not push `animl-analytics` to our repository as it was unneccessary
(we kept a local version). 

To download the SCI biodiversity dataset images run the following:
1. `cd animl-analytics`
2. `python utils/download_images.py --coco-file ../SCI_coco.json --output-dir ../images`  
  
Here, `../SCI_coco.json` is the path to our COCO json file for the SCI dataset, 
downloaded from Animl and `../images` is the path to the directory we would like
to store the images in. You can replace these with your own values. Note, the
COCO json file is only for images that are human reviewed and aren't labeled `empty`, 
`person`, `vehicle`,or `animal`. We did not upload the json file for privacy reasons.
  
If it takes too long to download the images/the session crashes in the middle, 
you can run the following script instead :  
  
`python utils/download_images_mod.py --coco-file ../SCI_coco.json --output-dir ../images`
  
The script `download_images_mod.py` is a modified version of `download_images.py` that
we created to download multiple images in parallel and skip already downloaded
images - this is faster than the original script. We have provided the 
`download_images_mod.py` file for your use. Make sure you move it into the 
`animl-analytics\utils` directory before running the above command.
  
After following the above steps, you should have a local folder containing the SCI
camera trap images. Note, this folder should not be pushed to the repository and
should persist locally. 

## How to create the Images-Labels dataset
The COCO json file for the SCI data contains three fields:
1. `images`: for each image, contains the `id` and `file_name`
2. `annotations`: for each image, contains the `image_id` matching `id` in 
`images`, the `category_id` corresponding to `id` in `categories` repersenting
the image's label, and `bbox` detailing the megadetector bounding box coordinates
3. `categories`: for each unique `category_id` in `annotations`, the corresponding
`name` referring to the species label
  
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


