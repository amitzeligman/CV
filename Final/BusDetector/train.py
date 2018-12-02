import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model import create_model
from dataLoader import data_generator_wrapper
import dataLoader
import os
from keras.utils import plot_model
from utils import get_anchors
from kmeans import YOLO_Kmeans


# Define main paths
projectDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
busDetectorDir = (os.path.dirname(os.path.abspath(__file__)))
dataDir = os.path.join(projectDir, "Data")
annotationsPath = os.path.join(dataDir, 'annotationsTrain.txt')
imagesDir = os.path.join(dataDir, "busesTrain")
logDir = os.path.join(busDetectorDir, 'logs/000/')

# Anchors
anchors = get_anchors('yolo_anchors.txt')
anchors = [10, 13,  16, 30,  33, 23,  30, 61,  62, 45,  59, 119,  116, 90,  156, 198,  373, 326]
anchors = np.array(anchors).reshape(-1, 2)

# Classes
classes = ['green', 'orange', 'white', 'silver', 'blue', 'red']
num_classes = len(classes)

input_shape = (3648, 2736)

# Hyper parameters

batch_size = 32
val_split = 0.1
learningRate = 1e-3


# Create model

model = create_model(input_shape, anchors, num_classes,
                     freeze_body=2, weights_path='model_data/yolo_weights.h5')  # make sure you know what you freeze

plot_model(model, to_file='model.png')
logging = TensorBoard(log_dir=logDir)
checkpoint = ModelCheckpoint(logDir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                             monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)


train_annotations, validation_annotations = dataLoader.get_meta_data(annotationsPath, val_split=val_split)

num_train = len(train_annotations)
num_val = len(validation_annotations)

# Train with frozen layers first, to get a stable loss.
# Adjust num epochs to your data set. This step is enough to obtain a not bad model.
if True:
    model.compile(optimizer=Adam(lr=learningRate), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(data_generator_wrapper(train_annotations, imagesDir,  batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator_wrapper(validation_annotations, imagesDir,  batch_size, input_shape, anchors,
                                                               num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[logging, checkpoint])
    model.save_weights(logDir + 'trained_weights_stage_1.h5')
