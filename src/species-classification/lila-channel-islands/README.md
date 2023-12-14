# Species Classification on LILA Channel Islands Dataset
This directory explores fine-tuning a pre-trained CNN classifier for species
classification on the LILA Channel Islands dataset, consisting of Channel Islands
camera trap images, as well as evaluating the model's classification results 
using GRAD-CAM. 

## Directory Structure
```
tnc-capstone/
├── src/
│   ├── species-classification/
│   │   ├── JDLP/... 
│   │   └── lila-channel-islands/
│   │   │   ├── README.md
│   │   │   ├── Dockerfile
│   │   │   ├── docker-shell.sh
│   │   │   ├── efficientNet.ipynb
│   │   │   ├── get_labels.py
│   │   │   ├── preprocess.py
│   │   │   ├── requirements.txt
│   │   │   ├── grad-cam.ipynb
│   │   │   └── lila_channel_islands_efficientnet_v1.png
```
A brief description of the files is given below:
* `docker-shell.sh`: create and run/activate the Docker container
* `Dockerfile`: blueprint for the Docker container
* `requirements.txt`: packages we want to install in the container
* `preprocess.py`: script for cropping images
* `get_labels.py`: script for creating a dataset connecting image file paths to
the corresponding species labels
* `efficientNet.ipynb`: notebook that does some EDA, fine-tunes EfficientNet on 
the Channel Islands images, and analyzes the model results
* `lila_channel_islands_efficientnet_v1.png`: confusion matrix image for model from `efficientNet.ipynb`
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

#### Preprocessing
The `preprocess.py` script is designed to automate the cropping of images in a dataset based on the bounding boxes predicted by the MegaDetector v5b model. It is intended to be used a single time to preprocess the dataset before training the species classification models. It utilizes the Google Cloud Storage client to read and write data from Google Cloud Storage buckets. The script is multi-threaded to improve processing efficiency.

    The script reads the JSON data from Google Cloud Storage. This JSON file contains information about the images and the bounding boxes detected by MegaDetector v5b.

    For each image in the dataset, the script checks if it has already been cropped. If not, it proceeds to crop the image based on the bounding boxes with a confidence level greater than 0.90. It then uploads the cropped image to the specified output folder in Google Cloud Storage.

### EfficientNetB0
The `efficientNet.ipynb` notebook initializes and configures the model architecture, making it ready for training.

- Dataset: [Channel Islands Camera Traps](https://lila.science/datasets/channel-islands-camera-traps/). This data set contains 246,529 camera trap images from 73 camera locations in the Channel Islands, California. All animals are annotated with bounding boxes. Animals are classified as rodent (82,914), fox (48,150), bird (11,099), skunk (1,071), or other (159). 114,949 images (47%) are empty.

  - Fine-tuning subset splits:
    <center>

      |      | Train | Val  | Test  |
      | ---- | ----- | ---- | ----- |
      | Size | 58,793 | 8,400 | 16,799 |
      | Proportion | 0.67 | 0.13 | 0.20 |

    </center>

  - Class distributions:
    <center>

    | Class  | Proportion in Split |
    | ------ | ------------------- |
    | fox    | 0.495               |
    | rodent | 0.430               |
    | bird   | 0.062               |
    | skunk  | 0.009               |
    | empty  | 0.003               |
    | other  | 0.001               |

    </center>
  
### Modelling 
- Model Architecture

    The model architecture consists of the following components:

    1. Base Model: A pre-trained EfficientNetB0 model is loaded without its top classification layers. This serves as the backbone of the classification model. The input shape is set to (224, 224, 3) to match the expected input size for the EfficientNetB0.

    2. Fine-tuning Layers: Global average pooling is applied to the output of the base model to reduce the spatial dimensions. A dense layer with a softmax activation function is added to perform the final classification. This predicts the probabilities of each class.

    3. Unfreezing Layers: The last 20 layers (excluding batch normalization layers) of the model are unfrozen for training. This allows the model to fine-tune its parameters on the specific dataset.

- Model training:
    The model was trained for 5 epochs, on a subset of the data, and the training and test sets were split 80/20. The model was evaluated on the test set, which was not used during training. The confusion matrix below shows the model's performance on the test set.

### Results: 
The fine-tuning of the EfficientNet model yielded overall high accuracy results across various classes. Notably, the Fox class achieved an outstanding average test accuracy of 99.3%, followed closely by the Rodent class with an accuracy of 99.6%. The Skunk and Bird classes also demonstrated strong performance, achieving accuracies of 95.3% and 96.9% respectively. However, the model faced challenges distinguishing between the 'Other' category, achieving an accuracy of 57.1%, and the 'Empty' category, where it achieved the lowest accuracy of 11.1%. This is likely due to the fact that the 'Other' category is comprised of a wide variety of species, and the 'Empty' category is comprised of images that do not contain any animals.  

<center>

![alt text](img/results/efficientNet.png "EfficientNetB0 Results")

| Class  | Average test accuracy |
| ------ | --------------------- |
| Fox    | 99.3%                 |
| Skunk  | 95.3%                 |
| Rodent | 99.6%                 |
| Bird   | 96.9%                 |
| Other  | 57.1%                 |
| Empty  | 11.1%                 |

</center>