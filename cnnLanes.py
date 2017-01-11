#!/home/pedgrfx/anaconda3/bin/python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from Camera import Camera
from moviepy.editor import VideoFileClip
from keras.models import model_from_json
import glob
import json
import csv
import cv2

image_sizeX = 160
image_sizeY = 80

def process_image(image):
    image = cv2.resize(image, (image_sizeX, image_sizeY))
    final_image = image[np.newaxis, ...]
    return final_image

def find_cnn_lane_eqs(img):
    test_image = process_image(img)
    prediction = model.predict(
                    x = test_image,
                    batch_size=1,
                    verbose=0)
    pred_ly2 = (prediction[0][0][0] * ly2Range) + ly2Min
    pred_ly =  (prediction[1][0][0] * lyRange) + lyMin
    pred_lc =  (prediction[2][0][0] * lcRange) + lcMin
    pred_ry2 = (prediction[3][0][0] * ry2Range) + ry2Min
    pred_ry =  (prediction[4][0][0] * ryRange) + ryMin
    pred_rc =  (prediction[5][0][0] * rcRange) + rcMin
    predList = (pred_ly2, pred_ly, pred_lc, pred_ry2, pred_ry, pred_rc)
    return predList

def draw_lanes(img, lane_eqs, Minv):
        # Create an image to draw the lines on
        color_img = np.zeros_like(img).astype(np.uint8)
    
        # Recast the x and y points into usable format for cv2.fillPoly()
        y = np.array(range(0,720))
        left_fitx = y*y*lane_eqs[0] + y*lane_eqs[1] + lane_eqs[2]
        left_fity = y
        right_fitx = y*y*lane_eqs[3] + y*lane_eqs[4] + lane_eqs[5]
        right_fity = y

        pts_left = np.array([np.transpose(np.vstack([left_fitx, left_fity]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_fity])))])
        pts = np.hstack((pts_left, pts_right))
    
        # Draw the lane onto the blank image
        cv2.fillPoly(color_img, np.int_([pts]), (0,255, 0))

        # Draw lane lines only if lane was detected this frame
        cv2.polylines(color_img, np.int_([pts_left]), False, (0,0,255), thickness=20)
        cv2.polylines(color_img, np.int_([pts_right]), False, (255,0,0), thickness=20)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_img, Minv, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        return result

def run_pipeline(img):
    # Apply distortion correction to the image
    undist = camera.undistort_image(img)

    #plt.imshow(undist)
    #plt.show()

    # Run CNN model on img
    lane_eqs = find_cnn_lane_eqs(undist)

    print("{}".format(lane_eqs))

    # Draw lines back onto road
    combined_img = draw_lanes(undist, lane_eqs, camera.Minv)

    #plt.imshow(combined_img)
    #plt.show()

    return combined_img


# Load the CNN model
with open("model.json", 'r') as jfile:
    model = model_from_json(json.load(jfile))

model.compile("adam", "mse")
weights_file = "model.h5"
model.load_weights(weights_file)

# Find values for normalizing
with open('lane_eqs.csv', 'r') as f:
    reader = csv.reader(f)
    driving_log_list = list(reader)
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

# Calibrate the camera
camera = Camera(num_x_points=9, num_y_points=6, debug_mode=False)
camera.calibrate_camera("/home/pedgrfx/SDCND/AdvancedLaneFinding/cnn/camera_cal/calibration*.jpg")

# Run our pipeline on the test video 
print("Running on test video1...")
clip = VideoFileClip("/home/pedgrfx/SDCND/AdvancedLaneFinding/cnn/project_video.mp4")
output_video = "/home/pedgrfx/SDCND/AdvancedLaneFinding/cnn/project_video_processed.mp4"
#clip = VideoFileClip("/home/pedgrfx/SDCND/AdvancedLaneFinding/cnn/challenge_video.mp4")
#output_video = "/home/pedgrfx/SDCND/AdvancedLaneFinding/cnn/challenge_video_processed.mp4"
#clip = VideoFileClip("/home/pedgrfx/SDCND/AdvancedLaneFinding/cnn/harder_challenge_video.mp4")
#output_video = "/home/pedgrfx/SDCND/AdvancedLaneFinding/cnn/harder_challenge_video_processed.mp4"
output_clip = clip.fl_image(run_pipeline)
output_clip.write_videofile(output_video, audio=False)
