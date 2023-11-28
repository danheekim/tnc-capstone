# Anomaly Detection
In this section we focus on exploring various anomaly detection methods on the Santa
Cruz Islands (SCI) dataset in Animl. 

## Directory structure

## How to download the SCI dataset from Animl
** YOU MUST HAVE AN ANIML ACCOUNT TO DO THE FOLLOWING **

First, make sure you are in the anomaly-detection directory. Then run the 
following in your terminal:
1. `pip install aws configure`
2. `pip install awscli`
3. `aws configure --profile animl`  
  
The third command above will prompt you for the following:
* AWS_ACCESS_KEY_ID
* AWS_SECRET_ACESS_KEY
* DEFAULT_REGION_NAME
* DEFAULT_OUTPUT_FORMAT
  
The AWS_ACESS_KEY_ID and AWS_SECRET_ACCESS_KEY were provided to us by Nathaniel
Rindlaub (nathaniel.rindlaub@tnc.org) - contact him for details. 
DEFAULT_REGION_NAME is `us-west-2` and DEFAULT_OUTPUT_FORMAT is `text`.

Once the above steps are completed, follow the instructions here: 
https://github.com/tnc-ca-geo/animl-analytics

This will create an `animl-analytics` directory in `anomaly-detection`. 
  
Here is the command we used to download the images (after following the 
instructions given in the above link and being in the `animl-analytics` 
directory):  
`python utils/download_images.py 
--coco-file ../SCI_coco.json 
--output-dir ../images`

If it takes too long to download the images/the session crashes in the middle, 
you can run the following:  
`python utils/download_images_mod.py 
--coco-file ../SCI_coco.json 
--output-dir ../images`
  
Where `download_images_mod.py` is a modified version of `download_images.py` that
we created to download multiple images in parallel and skip already downloaded
images - this is faster than the original script. We have provided the 
`download_images_mod.py` file make sure you move it into the 
`animl-analytics\utils` directory before running the above command.
  
This creates an `images` directory in the `anomaly-detection` folder which
contains the data.

Note, the `coco-file` specified above is our custom coco file for the SCI dataset
that filters out `empty`, `person`, `vehicle`, `animal` labeled images and only
contains human reviewed images. This means our dataset has a total of 110,320
images. Moreover, the `animl-analytics` and `images` directory are NOT pushed
to this GitHub; they exist locally. The `coco-file` is provided for 
reproducibility.  

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
into Google Colab (where we will be doing our exploration).   

The script for creating the dataset above is called `create_labels_df.py` and
can be run using the following command:    
`python create_labels_df.py`

