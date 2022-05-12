# Faster

Faster is a Python library for extracting features through the last layer of ResNet50 model

## Installation

```bash
git clone https://github.com/jashdalvi/Faster.git
pip install -r requirements.txt
```

## Usage


```python

from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
from faster import hdf5DatasetWriter,StoreFeatures
from imutils import paths



#Initializing the model
model = ResNet50(weights = "imagenet",include_top = False,input_shape = (224, 224, 3))

#Specifying the base directory of the dataset
#File organization example: 
# base_dir/class1
# base_dir/class2

base_dir = "/home/jash/Desktop/JashWork/Covid19CT/Dataset"

#Specifying the label mapping to integers
label_mapping = {"NonCOVID":0,"COVID":1}

#Initializing the store features and hdf5 dataset writer object
store_features = StoreFeatures(base_dir,model,label_mapping)
hdf5_obj = hdf5DatasetWriter("/home/jash/Desktop/JashWork/Covid19CT/features.hdf5",store_features.imagepaths,buffer_size = 64)

#store_features.storefeatures() will store all the extracted features in a hdf5 file
store_features.storefeatures(hdf5_obj,batch_size = 16)

```