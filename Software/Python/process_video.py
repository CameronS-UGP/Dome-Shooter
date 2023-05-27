# Cameron Shipman
# 
#
# This is the main video processing script.
# The main loop reads each frame from the mp4 file and then performs the detection
# of the yellow dot and reticle on it. The results are currently drawn on the frame
# and displayed.
# 
# The program requires 4 external libraries: OpenCV for Python, Matplotlib, tqdm and numpy.
# Note: Using pip to install OpenCV or Matplotlib will automatically install numpy if not present.
#
# Run like this: python process_video.py


import os
import subprocess
import reticle_finder as rf
import copy

try:
    import cv2
except ModuleNotFoundError as err:
    print("Please install cv2. E.g. Run 'pip install opencv-python' in a terminal.")       # This will install numpy if not present.
    # exit(1)

try:
    import numpy as np
except ModuleNotFoundError as err:
    print("Please install cv2. E.g. Run 'pip install numpy' in a terminal.")
    # exit(1)

try:
    from tqdm import tqdm
except ModuleNotFoundError as err:
    print("Please install cv2. E.g. Run 'pip install tqdm' in a terminal.")
    exit(1)

video_filepath = './langham_dome_0987.mp4'                                      # Created using convert_video.py.

# The frame of interest in the video. The rest show things like the count down
# before each session, or the aircraft information graphic etc.
# These frame numbers are when the screen is active and aircraft are approaching etc.
# Start   End           Yellow dots         Reticle?
#     1   410           1                   No
#   570   770           1                   No
#  1160  1860           1                   No      poor quality footage
#  2060  2865           1                   Yes
#  3335  3785           3                   Yes
#  3940  4355           3                   Yes
useful_frame_ranges = [(1, 410), (570, 770), (1160, 1860), (2060, 2865), (3335, 3785), (3940, 4355)]
# Subsets for development:
# useful_frame_ranges = [(2060, 2865), (3335, 3785), (3940, 4355)]
#useful_frame_ranges = [(2060, 2863)]
#useful_frame_ranges = [(570, 770)]
#useful_frame_ranges = [(3940, 4355)]
#useful_frame_ranges = [(4200, 4355)]


#-------------------------------------------------------------------------------
# Function to determine if a frame number is useful. I.e. is it in useful_frame_ranges.
# Return bool   - True if frame number is useful.
def frameNumberInRange(frame_number):
    for fr in useful_frame_ranges:
        if fr[0] <= frame_number <= fr[1]:
            return True
    return False


#-------------------------------------------------------------------------------
# Open the video file and obtain/print basic info.
#   video_file      String containing video file name/path.
# Returns:      cap, num_frames
def openVideoFile(video_file):
    assert(os.path.exists(video_file))

    # Load in video capture
    cap = cv2.VideoCapture(video_file)

    # Get video dimensions
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(f'Height     : {height}')
    print(f'Width      : {width}')

    # Get frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'FPS        : {fps:0.2f}')

    # Get number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Num frames : {num_frames}')

    return cap, num_frames


#-------------------------------------------------------------------------------
# Function to crop the image to eliminate some of the stuff around the screen.
# Ideally the video (or real-time camera feed) would be fixed and so the cropping
# could be much tighter.
# frame    - The frame image to crop.
# Returns cropped image
def cropFrame(frame):
    # Note that an image is a 3-dim mparray: comprising a list of rows, each row
    # is a list of colours, each colour is a tuple (e.g. B,G,R).
    # So the crop is therefore y1:y2, x1:x2.
    x1 = 300
    y1 = 200
    x2 = 1550
    y2 = 950
    return frame[y1:y2, x1:x2]          # w = 1250, h = 750


#-------------------------------------------------------------------------------
# Dump a frame.
# image             Cropped image to dump
# frame_number      Frame number of image.
def dumpFrame(image, frame_number):
    output_file = f'screen_area_{frame_number}.tiff'
    if os.path.exists(output_file):
        os.remove(output_file)
    print(f'Writing {output_file}...')
    cv2.imwrite(output_file, image)


#-------------------------------------------------------------------------------
# Dump every 10th frame starting at 2500. Used to build up a dataset of reticles.
def dumpFrames():
    cap, num_frame = openVideoFile(video_filepath)
    frame_number_step10 = 2500
    frame_number = 0
    while(cap.isOpened()):
        ret, frame_bgr = cap.read()
        if ret:
            frame_number += 1
        else:
            print(f'Number of frames processed: {frame_number} / {num_frames}')
            break

        if frame_number == frame_number_step10:
            screen_area_bgr = cropFrame(frame_bgr)
            dumpFrame(screen_area_bgr, frame_number)
            frame_number_step10 += 10


#-------------------------------------------------------------------------------
# Detect yellow dots.
# screen_area_hsv       HSV image to search.
# Return list of bounding boxes as (x, y, w, h) tuples.
def findYellowDots(screen_area_hsv):
    # Mask the image using the yellow dot colour range (HSV)
    # Hue: First value is 0 - 180, so Hue needs to be divided by 2.
    # Sat: Second value is 0-255, so Saturation percentage needs to be multiplied by 2.55.
    # Val: Third value is 0-255, so Value percentage needs to be multiplied by 2.55.
    lower_yellow_dot = np.array([20,50,160], dtype = "uint16")              # Hue 58, sat 26.7, value 73.3
    upper_yellow_dot = np.array([45,150,220], dtype = "uint16")             # Hue 67, sat 22.3, value 86.3          40 also worked.
    yellow_dot_mask = cv2.inRange(screen_area_hsv, lower_yellow_dot, upper_yellow_dot)
    #cv2.imshow('yellow_dot_mask', yellow_dot_mask)                         # For development.
    # Find the contours of all the yellow dot matches.
    contours, _ = cv2.findContours(yellow_dot_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Convert the contours to bounding boxes and eliminate any with out-of-range dimenions.
    yellow_dot_bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 10 <= w <= 30 and 10 <= h <= 30 and (w/h > 0.8 or h/w > 0.8):    # Bounding box must be 10-30 each side and 80% square.
            yellow_dot_bboxes.append((x, y, w, h ))

    return yellow_dot_bboxes


#-------------------------------------------------------------------------------
# Detect reticle
# screen_area_bgr       BGR image to search.
# Returns:              boolean, x, y
#   boolean     Was a reticle found?
#   x, y        Location of centre of reticle, if bFound = True
def findReticle(reticle_finder, screen_area_bgr):

    ''' Experiment
    # Remember cv2 is BGR       B   G   R
    #lower_reticle = np.array([216,223,211], dtype = "uint16")             # Hue 145, sat 5.4, value 87.5  ->  73, 14, 223
    #upper_reticle = np.array([218,225,213], dtype = "uint16")             # Hue 145, sat 5.3, value 88.2  ->  73, 14, 225
    lower_reticle = np.array([ 55, 10,180], dtype = "uint16")
    upper_reticle = np.array([ 80, 40,255], dtype = "uint16")
    reticle_mask = cv2.inRange(screen_area_hsv, lower_reticle, upper_reticle)
    #cv2.imshow('reticle_mask', reticle_mask)

    # Create a copy of the current image that has only a subset of the colours present, that the reticle uses.
    # All other colours set to average reticle colour.
    colour_reduced_img_bgr = copy.copy(screen_area_bgr)
    colour_reduced_img_bgr[reticle_mask==0]=(216,223,211)
    cv2.imshow('colour_reduced_img_bgr', colour_reduced_img_bgr)
    '''

    return reticle_finder.SearchForReticle(screen_area_bgr)


#-------------------------------------------------------------------------------
if __name__ == '__main__':

    #dumpFrames()            # One-off call to dump every 10th frame starting at 2500. So we can build up a dataset of reticles.
    #exit(0)


    reticle_finder = rf.ReticleFinder()

    cap, num_frames = openVideoFile(video_filepath)
    frame_number = 0
    progress_bar = tqdm(total=num_frames)                                       # Show a nice progress bar as we work through the video file.
    while(cap.isOpened()):                                                      # Use cap.isOpened() for extra robustness, even though we exit the loop if
                                                                                # cap.read() returns a false status value.
        ret, frame_bgr = cap.read()
        if ret:
            frame_number += 1
            progress_bar.update(1)
        else:
            print(f'Number of frames processed: {frame_number} / {num_frames}')
            break

        if frameNumberInRange(frame_number):

            # 1. Extract a smaller area to cover just the screen in the image -> screen
            # Approx for now
            screen_area_bgr = cropFrame(frame_bgr)
            #screen_area_bgr_blur = cv2.GaussianBlur(screen_area_bgr, (9, 9), 10)       # Gaussian experiments
            
            screen_area_hsv = cv2.cvtColor(screen_area_bgr, cv2.COLOR_BGR2HSV)
            #screen_area_hsv = cv2.cvtColor(screen_area_bgr_blur, cv2.COLOR_BGR2HSV)

            # Detect the yellow dots.
            yellow_dot_bboxes = findYellowDots(screen_area_hsv)

            # Detect reticle
            bFound, reticle_x, reticle_y  = findReticle(reticle_finder, screen_area_bgr)
            if bFound: #Draws circle around detected reticle
                cv2.circle(screen_area_bgr, (reticle_x, reticle_y), 1, (255,0,0), 1)
                cv2.circle(screen_area_bgr, (reticle_x, reticle_y), 200, (255,0,0), 1)

            # Draw rectangles around yellow dots (with 5 pixel margin), with a centre dot, in red.
            for ybb in yellow_dot_bboxes:
                x = ybb[0]
                y = ybb[1]
                w = ybb[2]
                h = ybb[3]
                cv2.circle(screen_area_bgr, (x + w//2, y + h//2), 1, (0,0,255), 1)
                cv2.rectangle(screen_area_bgr, (x - 5, y - 5), (x + w + 5, y + h + 5), (0,0,255), 1)

            cv2.imshow('screen_area_bgr', screen_area_bgr)
            #cv2.imshow('screen_area_bgr_blur', screen_area_bgr_blur)
            cv2.waitKey(1)
            #cv2.waitKey(0)
            #exit(0)

    cap.release()
    cv2.destroyAllWindows()
    progress_bar.close()


    reticle_finder.PrintMaxValGraph()                                           # Print reticle max_val graph.

    # All done.
