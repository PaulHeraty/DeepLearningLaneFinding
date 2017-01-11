#!/home/pedgrfx/anaconda3/bin/python

# This file generates a Keras mode (model.json) and a corresponding
# weights file (model.h5) which are used to implement behavioral cloning
# for driving a car around a race track. The model takes input frames
# (640x480x3) and labels which contain the steering angle for each frame.
# The model should then be able to predict a steering angle when presented
# which a previously un-seen frame. This can then be used to calculate how
# to steer a car on a track in order to stay on the road

################################################################
# Start by importing the required libraries
################################################################
import numpy as np
from keras.models import Model
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Input
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from random import shuffle
import scipy.stats as stats
import pylab as pl
import os
import cv2
import csv
import math
import json
from pandas.stats.moments import ewma
from keras.models import model_from_json


################################################################
# Define our variables here
################################################################
learning_rate = 0.0001  

image_sizeX = 160
image_sizeY = 80
num_channels = 3 
n_classes = 1 # This is a regression, not a classification
nb_epoch = 500
batch_size = 30
dropout_factor = 0.4
w_reg=0.0000

input_shape1 = (image_sizeY, image_sizeX, num_channels)
num_filters1 = 24
filter_size1 = 5
stride1=(2,2)
num_filters2 = 36
filter_size2 = 5
stride2=(2,2)
num_filters3 = 48
filter_size3 = 5
stride3=(2,2)
num_filters4 = 64
filter_size4 = 3
stride4=(1,1)
num_filters5 = 64
filter_size5 = 3
stride5=(1,1)
pool_size = (2, 2)
hidden_layers1 = 100
hidden_layers2 = 50

################################################################
# Define any functions that we need
################################################################

# Read in the image, re-size if necessary
def process_image(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, (image_sizeX, image_sizeY))
    final_image = image[np.newaxis, ...]
    return final_image

# Calculate the correct number of samples per epoch based on batch size
def calc_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    # return value must be a number than can be divided by batch_size
    samples_per_epoch = math.ceil((num_batches / batch_size) * batch_size)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch
    
# Import the training data
# Note: the training image data is stored in the IMG directory, and 
# are 640x480 RGB images. Since there will likely be thousands of these
# images, we'll need to use Python generators to access these, thus
# preventing us from running out of memory (which would happen if I 
# tried to store the entire set of images in memory as a list

def get_next_batch(image_list):
    index = 0
    while 1:
        final_images = np.ndarray(shape=(batch_size, image_sizeY, image_sizeX, num_channels), dtype=float)
        left_y2_coeff = np.ndarray(shape=(batch_size), dtype=float)
        left_y_coeff = np.ndarray(shape=(batch_size), dtype=float)
        left_c_coeff = np.ndarray(shape=(batch_size), dtype=float)
        right_y2_coeff = np.ndarray(shape=(batch_size), dtype=float)
        right_y_coeff = np.ndarray(shape=(batch_size), dtype=float)
        right_c_coeff = np.ndarray(shape=(batch_size), dtype=float)
        for i in range(batch_size):
            if index >= len(image_list):
                index = 0
                # Shuffle X_train after every epoch
                shuffle(image_list)
            filename = image_list[index][0]
            left_y2_val = (image_list[index][1] - ly2Min) / ly2Range
            left_y_val = (image_list[index][2] -lyMin) / lyRange
            left_c_val = (image_list[index][3] -lcMin) / lcRange
            right_y2_val = (image_list[index][4] -ry2Min) / ry2Range
            right_y_val = (image_list[index][5] -ryMin) / ryRange
            right_c_val = (image_list[index][6] -rcMin) / rcRange
            #print("Batch item {} :  {} {} {} {} {} {} {}".format(i, filename, left_y2_val, left_y_val, left_c_val, right_y2_val, right_y_val, right_c_val))
            final_image = process_image(filename)

            #final_angle = np.ndarray(shape=(1), dtype=float)
            #final_angle[0] = angle
            final_images[i] = final_image
            left_y2_coeff[i] = left_y2_val
            left_y_coeff[i] = left_y_val
            left_c_coeff[i] = left_c_val
            right_y2_coeff[i] = right_y2_val
            right_y_coeff[i] = right_y_val
            right_c_coeff[i] = right_c_val
            index += 1
        yield ({'img_in' : final_images}, {'left_y2_coeff' : left_y2_coeff, 'left_y_coeff' : left_y_coeff, 'left_c_coeff' : left_c_coeff, 'right_y2_coeff' : right_y2_coeff, 'right_y_coeff' : right_y_coeff, 'right_c_coeff' : right_c_coeff })

###############################################
############### START #########################
###############################################

# Start by reading in the .csv file which has the filenames and steering angles
# driving_log_list is a list of lists, where element [x][0] is the image file name
# and element [x][3] is the steering angle
with open('lane_eqs.csv', 'r') as f:
    reader = csv.reader(f)
    driving_log_list = list(reader)
num_frames = len(driving_log_list)
print("Found {} frames of input data.".format(num_frames))

# Process this list so that we end up with training images and labels
X_train = [("", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ) for x in range(num_frames)]
print(len(X_train))
for i in range(num_frames):
    X_train[i] = (driving_log_list[i][0].lstrip(),  # image
              float(driving_log_list[i][1]),  # left lane y^2 co-eff
              float(driving_log_list[i][2]),  # left lane y co-eff
              float(driving_log_list[i][3]),  # left lane c co-eff
              float(driving_log_list[i][4]),  # right lane y^2 co-eff
              float(driving_log_list[i][5]),  # right lane y co-eff
              float(driving_log_list[i][6]))  # right lane c co-eff

# Update num_frames as needed
num_frames = len(X_train)
print("After list pre-processing, now have {} frames".format(num_frames))

# Find values for normalizing
ly2List = [float(i[1]) for i in driving_log_list]
lyList = [float(i[2]) for i in driving_log_list]
lcList = [float(i[3]) for i in driving_log_list]
ry2List = [float(i[4]) for i in driving_log_list]
ryList = [float(i[5]) for i in driving_log_list]
rcList = [float(i[6]) for i in driving_log_list]
ly2Min = min(ly2List)
lyMin = min(lyList)
lcMin = min(lcList)
ry2Min = min(ry2List)
ryMin = min(ryList)
rcMin = min(rcList)
ly2Range = max(ly2List) - min(ly2List)
lyRange = max(lyList) - min(lyList)
lcRange = max(lcList) - min(lcList)
ry2Range = max(ry2List) - min(ry2List)
ryRange = max(ryList) - min(ryList)
rcRange = max(rcList) - min(rcList)

# Split some of the training data into a validation dataset.
# First lets shuffle the dataset
shuffle(X_train)
num_train_elements = int((num_frames/4.)*3.)
num_valid_elements = int(((num_frames/4.)*1.) / 2.)
X_valid = X_train[num_train_elements:num_train_elements + num_valid_elements]
X_test = X_train[num_train_elements + num_valid_elements:]
X_train = X_train[:num_train_elements]
print("X_train has {} elements.".format(len(X_train)))
print("X_valid has {} elements.".format(len(X_valid)))
print("X_test has {} elements.".format(len(X_test)))

################################################################
# Build a new CNN Network with Keras
################################################################
#model = Sequential()
img_in = Input(shape = input_shape1, name='img_in')
# CNN Layer 1
norm_l1 = BatchNormalization(input_shape = input_shape1, axis=1)(img_in)
cnn_l1 = Convolution2D(nb_filter=num_filters1, 
                    nb_row=filter_size1, 
                    nb_col=filter_size1,
                    subsample=stride1,
                    border_mode='valid',
                    input_shape=input_shape1, 
                    W_regularizer=l2(w_reg))(norm_l1)
cnn_r1 = Activation('relu')(cnn_l1)
# CNN Layer 2
cnn_l2 = Convolution2D(nb_filter=num_filters2, 
                    nb_row=filter_size2, 
                    nb_col=filter_size2,
                    subsample=stride2,
                    border_mode='valid', 
                    W_regularizer=l2(w_reg))(cnn_r1)
cnn_d2 = Dropout(dropout_factor)(cnn_l2)
cnn_r2 = Activation('relu')(cnn_d2)
# CNN Layer 3
cnn_l3 = Convolution2D(nb_filter=num_filters3, 
                    nb_row=filter_size3, 
                    nb_col=filter_size3,
                    subsample=stride3,
                    border_mode='valid', 
                    W_regularizer=l2(w_reg))(cnn_r2)
cnn_d3 = Dropout(dropout_factor)(cnn_l3)
cnn_r3 = Activation('relu')(cnn_d3)
# CNN Layer 4
cnn_l4 = Convolution2D(nb_filter=num_filters4,
                    nb_row=filter_size4, 
                    nb_col=filter_size4,
                    subsample=stride4,
                    border_mode='valid', 
                    W_regularizer=l2(w_reg))(cnn_r3)
cnn_d4 = Dropout(dropout_factor)(cnn_l4)
cnn_r4 = Activation('relu')(cnn_d4)
# CNN Layer 5
cnn_l5 = Convolution2D(nb_filter=num_filters5,
                    nb_row=filter_size5, 
                    nb_col=filter_size5,
                    subsample=stride5,
                    border_mode='valid', 
                    W_regularizer=l2(w_reg))(cnn_r4)
cnn_d5 = Dropout(dropout_factor)(cnn_l5)
cnn_r5 = Activation('relu')(cnn_d5)
# Flatten
flat = Flatten()(cnn_r5)
############## left_y2_coeff ##########################
# FCNN Layer 1
ly2_d1 = Dense(hidden_layers1, input_shape=(2496,), name="left_y2_hidden1", W_regularizer=l2(w_reg))(flat)
ly2_r1 = Activation('relu')(ly2_d1)
# FCNN Layer 2
ly2_d2 = Dense(hidden_layers2, name="left_y2_hidden2", W_regularizer=l2(w_reg))(ly2_r1)
ly2_r2 = Activation('relu')(ly2_d2)
# FCNN Layer 3
left_y2_coeff = Dense(n_classes, name="left_y2_coeff", W_regularizer=l2(w_reg))(ly2_r2)
############## left_y_coeff ##########################
# FCNN Layer 1
ly_d1 = Dense(hidden_layers1, input_shape=(2496,), name="left_y_hidden1", W_regularizer=l2(w_reg))(flat)
ly_r1 = Activation('relu')(ly_d1)
# FCNN Layer 2
ly_d2 = Dense(hidden_layers2, name="left_y_hidden2", W_regularizer=l2(w_reg))(ly_r1)
ly_r2 = Activation('relu')(ly_d2)
# FCNN Layer 3
left_y_coeff = Dense(n_classes, name="left_y_coeff", W_regularizer=l2(w_reg))(ly_r2)
############## left_c_coeff ##########################
# FCNN Layer 1
lc_d1 = Dense(hidden_layers1, input_shape=(2496,), name="left_c_hidden1", W_regularizer=l2(w_reg))(flat)
lc_r1 = Activation('relu')(lc_d1)
# FCNN Layer 2
lc_d2 = Dense(hidden_layers2, name="left_c_hidden2", W_regularizer=l2(w_reg))(lc_r1)
lc_r2 = Activation('relu')(lc_d2)
# FCNN Layer 3
left_c_coeff = Dense(n_classes, name="left_c_coeff", W_regularizer=l2(w_reg))(lc_r2)
############## right_y2_coeff ##########################
# FCNN Layer 1
ry2_d1 = Dense(hidden_layers1, input_shape=(2496,), name="right_y2_hidden1", W_regularizer=l2(w_reg))(flat)
ry2_r1 = Activation('relu')(ry2_d1)
# FCNN Layer 2
ry2_d2 = Dense(hidden_layers2, name="right_y2_hidden2", W_regularizer=l2(w_reg))(ry2_r1)
ry2_r2 = Activation('relu')(ry2_d2)
# FCNN Layer 3
right_y2_coeff = Dense(n_classes, name="right_y2_coeff", W_regularizer=l2(w_reg))(ry2_r2)
############## left_y_coeff ##########################
# FCNN Layer 1
ry_d1 = Dense(hidden_layers1, input_shape=(2496,), name="right_y_hidden1", W_regularizer=l2(w_reg))(flat)
ry_r1 = Activation('relu')(ry_d1)
# FCNN Layer 2
ry_d2 = Dense(hidden_layers2, name="right_y_hidden2", W_regularizer=l2(w_reg))(ry_r1)
ry_r2 = Activation('relu')(ry_d2)
# FCNN Layer 3
right_y_coeff = Dense(n_classes, name="right_y_coeff", W_regularizer=l2(w_reg))(ry_r2)
############## left_c_coeff ##########################
# FCNN Layer 1
rc_d1 = Dense(hidden_layers1, input_shape=(2496,), name="right_c_hidden1", W_regularizer=l2(w_reg))(flat)
rc_r1 = Activation('relu')(rc_d1)
# FCNN Layer 2
rc_d2 = Dense(hidden_layers2, name="right_c_hidden2", W_regularizer=l2(w_reg))(rc_r1)
rc_r2 = Activation('relu')(rc_d2)
# FCNN Layer 3
right_c_coeff = Dense(n_classes, name="right_c_coeff", W_regularizer=l2(w_reg))(rc_r2)

model = Model(input=[img_in], output=[left_y2_coeff, left_y_coeff, left_c_coeff, right_y2_coeff, right_y_coeff, right_c_coeff])

model.summary()

################################################################
# Train the network using generators
################################################################
print("Number of Epochs : {}".format(nb_epoch))
print("  Batch Size : {}".format(batch_size))
print("  Training batches : {} ".format(calc_samples_per_epoch(len(X_train), batch_size)))
print("  Validation batches : {} ".format(calc_samples_per_epoch(len(X_valid), batch_size)))

print("*** Compiling new model wth learning rate {} ***".format(learning_rate))

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=learning_rate)
              )

history = model.fit_generator(
                    get_next_batch(X_train),	# The generator to return batches to train on
                    nb_epoch=nb_epoch,  		# The number of epochs we will run for
                    max_q_size=10,      		# Max generator items that are queued and ready 
                    samples_per_epoch=calc_samples_per_epoch(len(X_train), batch_size),
                    validation_data=get_next_batch(X_valid),	# validation data generator
                    nb_val_samples=calc_samples_per_epoch(len(X_valid), batch_size),
                    verbose=1)

# Evaluate the accuracy of the model using the test set
score = model.evaluate_generator(
                    generator=get_next_batch(X_test),	# validation data generator
                    val_samples=calc_samples_per_epoch(len(X_test), batch_size), # How many batches to run in one epoch
                    )
print("Test score {}".format(score))


# Predict on a given image
score = model.predict( process_image("./video_frames/frame0.jpg"), batch_size=1)

################################################################
# Save the model and weights
################################################################
model_json = model.to_json()
with open("./model.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights("./model.h5")
print("Saved model to disk")
