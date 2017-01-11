import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


class Camera:
    def __init__(self, num_x_points, num_y_points, debug_mode=False):
        # Number of x points in test images
        self.num_x_points = num_x_points
        # Number of y points in test images
        self.num_y_points = num_y_points
        # Camera Matrix
        self.mtx = 0
        # Camera Distortion coefficients
        self.dist = 0
        # Camera rotation vectors
        self.rvecs = 0 
        #Camera translation vectors
        self.tvecs = 0
        # Debug mode
        self.debug_mode = debug_mode
        # Source coords for perspective xform
        self.src_coords = np.float32([[240,719],
                         [579,450],
                         [712,450],
                         [1165,719]])
        # Dest coords for perspective xform
        self.dst_coords = np.float32([[300,719],
                         [300,0],
                         [900,0],
                         [900,719]])
        # Perspective Transform matrix
        self.M = cv2.getPerspectiveTransform(self.src_coords, self.dst_coords)
        # Inverse Perspective Transform matrix
        self.Minv = cv2.getPerspectiveTransform(self.dst_coords, self.src_coords)
        
    def set_debug_mode(self, mode):
        self.debug_mode = mode

    def calibrate_camera(self, files):
        print("Calibrating camera...")
        images = glob.glob(files)

        objpoints = [] # 3D points in real world space
        imgpoints = [] # 2S points in image plane

        # Prepare objpoints like (0,0,0), (1,0,0), ... (7,5,0)
        objp = np.zeros((self.num_y_points*self.num_x_points,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.num_x_points,0:self.num_y_points].T.reshape(-1,2)

        # Read in images and find chessboard corners
        for fname in images:
            # Read in 
            img = mpimg.imread(fname)
            # Convert image to gray
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.num_x_points, self.num_y_points), None)

            # If corners are found, add object points, image points
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)

                # draw and display the corners
                img = cv2.drawChessboardCorners(img, (self.num_x_points, self.num_y_points), corners, ret)
            else:
                print("    Warning: Could not find correct number of corners for image {}".format(fname))

        # Get camera calibration params
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    def undistort_image(self, img):
        # Apply distortion correction to the image
        undist_img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        return undist_img

    def threshold_binary(self, img):
        if self.debug_mode:
            plt.imshow(img)
            plt.title('Orig Image')
            plt.show()
        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        if self.debug_mode:
            plt.imshow(s_channel)
            plt.title('S Channel')
            plt.show()

        # Grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        if self.debug_mode:
            plt.imshow(scaled_sobel)
            plt.title('Sobel X')
            plt.show()

        # Threshold x gradient
        thresh_min = 30
        thresh_max = 150
        sxbinary = np.zeros_like(scaled_sobel)
        #sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        retval, sxthresh = cv2.threshold(scaled_sobel, 30, 150, cv2.THRESH_BINARY)
        sxbinary[(sxthresh >= thresh_min) & (sxthresh <= thresh_max)] = 1
        if self.debug_mode:
            plt.imshow(sxthresh)
            plt.title('sxthresh')
            plt.show()
            plt.imshow(sxbinary)
            plt.title('Threshold Gradient')
            plt.show()

        # Threshold color channel
        s_thresh_min = 175
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        # Use inRange instead of multiple thresholds
        #retval, s_thresh1 = cv2.threshold(s_channel.astype('uint8'), 175, 255, cv2.THRESH_BINARY)    
        #retval, s_thresh2 = cv2.threshold(s_channel.astype('uint8'), 250, 255, cv2.THRESH_BINARY)    
        #s_thresh = np.zeros_like(s_binary)
        #s_thresh = s_thresh1 - s_thresh2
        #s_binary[(s_thresh >= s_thresh_min) & (s_thresh <= s_thresh_max)] = 1
        s_thresh = cv2.inRange(s_channel.astype('uint8'), 175, 250)

        s_binary[(s_thresh == 255)] = 1
        if self.debug_mode:
            plt.imshow(s_thresh)
            plt.title('s_thresh')
            plt.show()
            plt.imshow(s_binary)
            plt.title('Threshold Binary')
            plt.show()
      
        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        if self.debug_mode:
            color_binary = np.dstack(( np.zeros_like(sxthresh), sxthresh, s_thresh))
            plt.imshow(color_binary)
            plt.title('Color Binary')
            plt.show()

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        # Plotting thresholded images
        if self.debug_mode:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.set_title('Stacked thresholds')
            ax1.imshow(color_binary)
    
            ax2.set_title('Combined S channel and gradient thresholds')
            ax2.imshow(combined_binary, cmap='gray')
            plt.show()

        return combined_binary

    def perspective_transform(self, img):
        img_size = (img.shape[1], img.shape[0])
        perspective_img = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)
        return perspective_img
