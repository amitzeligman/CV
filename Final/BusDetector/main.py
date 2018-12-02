import dataLoader
import os
import utils
import numpy as np


# Define main paths
projectDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
busDetectorDir = (os.path.dirname(os.path.abspath(__file__)))
dataDir = os.path.join(projectDir, "Data")
annotationsPath = os.path.join(dataDir, 'annotationsTrain.txt')
imagesDir = os.path.join(dataDir, "busesTrain")


input_shape = (2736, 3648)
# Anchors
anchors = [10, 13,  16, 30,  33, 23,  30, 61,  62, 45,  59, 119,  116, 90,  156, 198,  373, 326]
anchors = np.array(anchors).reshape(-1, 2)

# Classes
classes = ['green', 'orange', 'white', 'silver', 'blue', 'red']
num_classes = len(classes)

train_annotations, validation_annotations = dataLoader.get_meta_data(annotationsPath)

# dataLoader.get_random_data(validation_annotations[0], input_shape, imagesDir)


x = dataLoader.data_generator(validation_annotations, imagesDir, 3, input_shape, anchors, num_classes)




