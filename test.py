#!/home/pedgrfx/anaconda3/bin/python
import argparse
import base64
import json
import csv
import cv2

import numpy as np
#import socketio
#import eventlet
#import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import matplotlib.pyplot as plt

def graph(x, p):
    y = np.array(range(0,720))
    left_lane_x = y*y*x[1] + y*x[2] + x[3]
    plt.plot(left_lane_x, y, 'b')
    left_lane_pred_x = y*y*p[0] + y*p[1] + p[2]
    plt.plot(left_lane_pred_x, y, 'r')

    right_lane_x = y*y*x[4] + y*x[5] + x[6]
    plt.plot(right_lane_x, y, 'b')
    right_lane_pred_x = y*y*p[3] + y*p[4] + p[5]
    plt.plot(right_lane_pred_x, y, 'r')
    plt.show()

image_sizeX = 160
image_sizeY = 80

with open("model.json", 'r') as jfile:
    model = model_from_json(json.load(jfile))

model.compile("adam", "mse")
weights_file = "model.h5"
model.load_weights(weights_file)

with open('lane_eqs.csv', 'r') as f:
    reader = csv.reader(f)
    driving_log_list = list(reader)
num_frames = len(driving_log_list)

X_train = [("", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ) for x in range(num_frames)]
for i in range(num_frames):
    X_train[i] = (driving_log_list[i][0].lstrip(),  # image
              float(driving_log_list[i][1]),  # left lane y^2 co-eff
              float(driving_log_list[i][2]),  # left lane y co-eff
              float(driving_log_list[i][3]),  # left lane c co-eff
              float(driving_log_list[i][4]),  # right lane y^2 co-eff
              float(driving_log_list[i][5]),  # right lane y co-eff
              float(driving_log_list[i][6]))  # right lane c co-eff

x_test = X_train[0]  
print("Original values : {}".format(x_test))

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


def process_image(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, (image_sizeX, image_sizeY))
    final_image = image[np.newaxis, ...]
    return final_image

# Test the model on an image and see how well it performs vs. the label
test_image = process_image(x_test[0])
prediction = model.predict(
                    x = test_image,
                    batch_size=1,
                    verbose=1)

pred_ly2 = (prediction[0][0][0] * ly2Range) + ly2Min
pred_ly =  (prediction[1][0][0] * lyRange) + lyMin
pred_lc =  (prediction[2][0][0] * lcRange) + lcMin
pred_ry2 = (prediction[3][0][0] * ry2Range) + ry2Min
pred_ry =  (prediction[4][0][0] * ryRange) + ryMin
pred_rc =  (prediction[5][0][0] * rcRange) + rcMin
predList = (pred_ly2, pred_ly, pred_lc, pred_ry2, pred_ry, pred_rc)
print("Model predicted {}".format(predList))

graph(x_test, predList)
