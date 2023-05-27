# Cameron Shipman
# 
#
# This script provides reticle finding functions.
# Create an instance of the ReticleFinder class (the constructor calls ExtractTemplate).
# Then call the SearchForReticle method for each image to get the reticle location, if found.
#
# Run like this: python reticle_finder.py

import os
import shutil
import copy

import dataset_helper as dh     # For Label class etc.

try:
    import cv2
except ModuleNotFoundError as err:
    print("Please install cv2. E.g. Run 'pip install opencv-python' in a terminal.")       # This will install numpy if not present.

try:
    import numpy as np
except ModuleNotFoundError as err:
    print("Please install cv2. E.g. Run 'pip install numpy' in a terminal.")

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as err:
    print("Please install cv2. E.g. Run 'pip install matplotlib' in a terminal.")           # This will install numpy if not present.
    exit(1)

plt.style.use('ggplot')

class ReticleFinder:

    def __init__(self):
        self.template_image = None
        self.reticle_mask = None
        self.max_vals = []
        self.ExtractTemplate()

    #-------------------------------------------------------------------------------
    # Extract the template (and mask?) from the chosen file. This is basically a
    # cropping exercise, using the bounding box info found in the accomplaning label
    # text file.
    # Return the template image.        todo: Save this to a file, once stable, and read it in.
    def ExtractTemplate(self):

        # Create a reticle template
        self.template_image = self.ExtractLabelledArea('Snapshots/screen_area_4020.tiff')

        # Show template to be sure.
        #cv2.imshow('template', self.template_image)
        #cv2.waitKey(0)

        # Create a reticle mask
        mask_image_bgr = self.ExtractLabelledArea('Snapshots_reticle_on_dark_screen/screen_area_2870.tiff')

        # Show template to be sure.
        #cv2.imshow('mask_image_bgr', mask_image_bgr)
        #cv2.waitKey(0)

        # Create a mask from the image.
        mask_image_hsv = cv2.cvtColor(mask_image_bgr, cv2.COLOR_BGR2HSV)
        lower_reticle = np.array([130,70,120], dtype = "uint16")
        upper_reticle = np.array([255,111,205], dtype = "uint16")
        self.reticle_mask = cv2.inRange(mask_image_hsv, lower_reticle, upper_reticle)
        cv2.imshow('reticle_mask', self.reticle_mask)                                # For development.
        cv2.waitKey(0)


    #-------------------------------------------------------------------------------
    # Extract the bounded area from the image file. This is basically a
    # cropping exercise, using the bounding box info found in the accomplaning label
    # text file.
    #   image_filename      Name of image file to extract from.
    # Return the cropped image.
    def ExtractLabelledArea(self, image_filename):

        #
        # Create a reticle template
        #
        # Read in a example image with a label text file defined.
        image = cv2.imread(image_filename)

        # Read in the bounding box file.
        labels = dh.ReadUnnormalisedLabelFile(dh.ConvertImageFilenameToLabelFilename(image_filename))
        assert(len(labels) == 1)        # Only expecting one label
        label = labels[0]

        # Extract just the bounding box. This is the template.
        x1 = label.x_centre - label.width // 2
        y1 = label.y_centre - label.height // 2
        x2 = x1 + label.width
        y2 = y1 + label.height
        cropped_image = image[y1:y2, x1:x2]

        # Temp for development:
        image_copy = copy.copy(image)
        cv2.circle(image_copy, (label.x_centre, label.y_centre), 1, (255,0,0), 1)
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (255,0,0), 1)
        cv2.imshow(image_filename, image_copy)
        cv2.waitKey(0)

        return cropped_image


    #-------------------------------------------------------------------------------
    def SearchForReticle(self, image):

        # Find subset of colours in image.


        #
        # Use a sliding window over the image to find correlation results.
        #
        # Luckily cv2 has just what we need and a selection of formulae.
        # TM_CCOEFF_NORMED works some of the time. With a threshold of 0.69, there are minimal false positives, but it does struggle with false negatives.
        #
        #match_result = cv2.matchTemplate(image, self.template_image, cv2.TM_SQDIFF, mask=???)
        #match_result = cv2.matchTemplate(image, self.template_image, cv2.TM_SQDIFF_NORMED, mask=???)
        #match_result = cv2.matchTemplate(image, self.template_image, cv2.TM_CCORR, mask=self.reticle_mask)
        #match_result = cv2.matchTemplate(image, self.template_image, cv2.TM_CCORR_NORMED, mask=???)
        #match_result = cv2.matchTemplate(image, self.template_image, cv2.TM_CCOEFF, mask=self.reticle_mask)
        match_result = cv2.matchTemplate(image, self.template_image, cv2.TM_CCOEFF_NORMED)  # Cannot take a mask
        cv2.imshow('match_result', match_result)
        #cv2.waitKey(0)

        #
        # Choose highest correlation and display it
        #
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
        #print(min_val)
        # print(max_val)
        #print(min_loc)
        #print(max_loc)
        self.max_vals.append(max_val)

        # Apply threshold.
        bFound = max_val >= 0.69
        h, w, _ = self.template_image.shape
        x, y = max_loc
        return bFound, x + w//2, y + h//2             # return centre


    def PrintMaxValGraph(self):
        plt.hist(self.max_vals, bins=30)
        plt.show()


if __name__ == '__main__':

    reticle_finder = ReticleFinder()

    #
    # Read in an example image
    #
    # Use image_with_template for now.
    image_filename = 'Snapshots/screen_area_4030.tiff'
    image = cv2.imread(image_filename)

    bFound, x, y = reticle_finder.SearchForReticle(image)

    assert(bFound)

    cv2.rectangle(image, (x-200, y-200), (x+200, y+200), (255, 0, 0), 2)
    cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
    cv2.imshow(f'Processed {image_filename}', image)
    cv2.waitKey(0)

