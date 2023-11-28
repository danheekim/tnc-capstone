# Species Classification on LILA Channel Islands Dataset
This directory explores fine-tuning a pre-trained CNN classifier for species
classification on the LILA Channel Islands dataset, consisting of Channel Islands
camera trap images, as well as evaluating the model's classification results 
using GRAD-CAM. 

## Directory Structure
------------
      └── lila-channel-islands
            ├── README.md
            ├── docker-shell.sh
            ├── Dockerfile
            ├── requirements.txt
            ├── preprocess.py
            ├── get_labels.py
            ├── efficientNet.ipynb
            └── grad-cam.ipynb      
            
--------
A brief description of the files is given below:
* `docker-shell.sh`: create and run/activate the Docker container
* `Dockerfile`: blueprint for the Docker container
* `requirements.txt`: packages we want to install in the container
* `preprocess.py`: script for cropping images
* `get_labels.py`: script for creating a dataset connecting image file paths to
the corresponding species labels
* `efficientNet.ipynb`: notebook that does some EDA, fine-tunes EfficientNet on 
the Channel Islands images, and analyzes the model results
* `grad-cam.ipynb`: notebook for using GRAD-CAM to analyze/interpret the fine-tuned EfficientNet
classification results

## Getting & Preprocessing the Data
The LILA Channel Islands dataset can be found here:
https://lila.science/datasets/channel-islands-camera-traps/
  
To store the data, we used GCP. To do this we had to create a service account.
Note, this approach costs money. If you run out of money, you won't be able
to access what you stored in the storage bucket (our current issue).   

The first step is to transfer the LILA Channel Islands dataset stored in GCP into
your own bucket. You can do this by creating a bucket for your service account and
using the GCP UI to transfer the data. Refer to the website linked above for details.
This will create an `images` folder in your bucket with the LILA Channel Islands images.  

Next, make sure you also have the following in your bucket (can manually upload):  
* LILA Channel Islands MegaDetector bounding box results json which can be found here: https://lila.science/megadetector-results-for-camera-trap-datasets/ 
* True labels csv for LILA datasets: https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/
  
For modeling purposes, we don't want to use the raw LILA Channel Islands images.
We need to crop them to their MegaDetector-predicted bounding boxes. To crop the 
images and store them in another folder in your GCP bucket:    
1. Create a `secrets` directory containing your service account json key file
2. Make the appropriate changes (as outlined by the comments) in the following 
files: `docker-shell.sh`, `preprocess.py`, `get_labels.py`
3. Activate the custom docker container enabeled with your GCP credentials by running
this command: `sh docker-shell.sh`
4. Crop the LILA Channel Islands images to their bounding boxes and store them in
a new folder in your bucket by running this command: `python preprocess.py`  

## Creating Images-Labels Intermediary Dataframe for Modeling
To make our lives easier, we created a dataframe that has the path to each cropped
image in our GCP bucket and the species (label) of that image. We can then use
this dataset to load in the images and labels during modeling. To create this 
dataframe and to save it as a csv in your bucket run the following command:  

`python get_labels.py`  

