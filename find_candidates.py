# Imports
import numpy as np
import os
import cv2 as cv
import time
from is_right import is_right
from remove_background import remove_background


def show_image(img_to_show, img_name, factor=1.0):

    """Function to display an image with a specific window name and specific resize factor

    Parameter:
        image(numpy array): Original image.
        img_name(string): Name for the window of the image
        factor(float): Resize factor (between 0 and 1)

   """
    num_rows = img_to_show.shape[0]
    num_cols = img_to_show.shape[1]
    img_to_show = cv.resize(img_to_show, None, fx=factor, fy=factor)
    cv.imshow(img_name, img_to_show)


def bring_to_256_levels(the_image):

    """Function normalize image to 0-255 range (8 bits)

    Parameter:
        image(numpy array): Original image.

    Return:
        img_new(numpy array, uint8): Normalized image.

   """
    if(the_image.max() == the_image.min()):
        return the_image.astype(np.uint8)
    img_as_double = the_image.astype(float)
    normalized = np.divide((img_as_double - np.amin(img_as_double)), (np.amax(img_as_double) - np.amin(img_as_double)))
    normalized = normalized*(pow(2, 8) - 1)
    return normalized.astype(np.uint8)


def getLinearSE(size, angle):

    """Function to create a linear SE with a specific size and angle.

    Parameter:
        size(int): Size of the square that contains the linear SE
        angle(int): Number that identifies an angle for the linear SE according to following options
                    1: 0°
                    2: 22.5°
                    3: 45°
                    4: 67.5°
                    5: 90°
                    6: 112.5
                    7: 135°
                    8: 157.5°
                    9: 11.25°
                    10: 78.75°
                    11: 101.25°
                    12: 168.75

    Returns:
        SE_diff (Numpy array): Binary array of size <size> that contains linear SE with approximate angle given by <angle>

   """

    if angle==1 or angle == 5:
        SE_horizontal = np.zeros((size, size))
        SE_horizontal[int((size - 1) / 2), :] = np.ones((1, size))
        if angle==1:
            return SE_horizontal.astype(np.uint8)
        else: #If vertical
            return np.transpose(SE_horizontal).astype(np.uint8)
    elif angle == 3 or angle == 7: #If 45 or 135
        SE_diagonal = np.eye(size)
        if angle == 3:
            return np.fliplr(SE_diagonal).astype(np.uint8)
        else:
            return SE_diagonal.astype(np.uint8)
    elif angle in [2,4,6,8]: #Angle more comples
        SE_diff = np.zeros((size, size))
        row = int(((size-1)/2)/2)
        col = 0
        ctrl_var = 0
        for i in range(size):
            if ctrl_var == 2:
                row = row +1
                ctrl_var = 0
            SE_diff[row, col] = 1
            col=col+1
            ctrl_var = ctrl_var + 1
        if angle == 8:
            return SE_diff.astype(np.uint8)
        elif angle == 2:
            return np.flipud(SE_diff).astype(np.uint8)
        elif angle == 4:
            return np.fliplr(np.transpose(SE_diff)).astype(np.uint8)
        else:
            return np.transpose(SE_diff).astype(np.uint8)
    elif angle in [9,10,11,12]:
        SE_diff = np.zeros((size, size))
        row = int(((size-1)/2)/2) + int( ((((size-1)/2)/2)-1)/2 )
        col = 0
        ctrl_var = 0
        for i in range(size):
            if ctrl_var == 3:
                row = row + 1
                ctrl_var = 0
            SE_diff[row, col] = 1
            col = col + 1
            ctrl_var = ctrl_var + 1
        if angle == 9:
            return np.flipud(SE_diff).astype(np.uint8)
        elif angle == 10:
            return np.fliplr(np.transpose(SE_diff)).astype(np.uint8)
        elif angle == 11:
            return np.transpose(SE_diff).astype(np.uint8)
        else:
            return SE_diff.astype(np.uint8)


def discard_regions_processing(the_image, connectivity, d1, d2):

    """Function to discard regions whose area is greater than the one specified by  the range [(pi/4)*d1^2, 1.3*(pi/4)*d2^2].

    Parameter:
        image(numpy array): Original binary image.
        connectivity(int): Connectivity for the pixels
        d1(int): Minimum diameter for the regions
        d2(int): Maximum diameter for the regions

    Return:
        output_image(numpy array): Image without regions that have an area that lies outside the mentioned range.

   """

    correction_factor = 1.3 #Factor for correcting area limits (because linear elements do not form a circle)
    output_image = np.zeros(the_image.shape)
    #Find connected components
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(the_image, connectivity)
    area_range = np.array([(np.pi/4)*pow(d1,2), (np.pi/4)*pow(d2,2)])
    for i_label in range(nlabels):
        #Check area range
        if(stats[i_label, 4]>= area_range[0] and stats[i_label, 4]<= area_range[1]):
            #Check bounding box dimensions
            if(stats[i_label, 2] >= correction_factor*d2 or stats[i_label, 3] >= correction_factor*d2):
                continue
            else:
                output_image[labels==i_label] = 1
    return output_image


def expand_image(the_image):

    """Function to expand image so that result is not affected during processing with linear SEs. The function simply replicates pixels of the flat part of the image to expand it.

    Parameter:
        the_image(numpy array): Input image to be expanded.

    Return:
        expanded_image(numpy array): Output image after expansion.
        cols_to_add(int): Number of columns that have to be added (equal to the 1/3 of the width of the image).
        offset(int): Number of black columns in the image from the border to the flat part of the mammogram tissue representation.
        side(string): Either "left" or "right" depending on the location of the breast in the image.

   """
    num_rows = the_image.shape[0]
    num_cols = the_image.shape[1]
    #Define number of columns to add to the image
    cols_to_add = int(num_cols / 3)
    side= "right"
    offset = 0
    #Create array for output image
    expanded_image = np.zeros((num_rows, num_cols + cols_to_add))
    #Expand image depending on the side of the breast in the image
    if(is_right(the_image)): #Breast on the right
        #Search for first column with non-zero values (some images have black column before breast tissue)
        offset = 0
        for i_col in range(num_cols):
            curr_col = the_image[:, -i_col-1]
            if(curr_col.sum()/num_rows > 80):
                offset = i_col+1
                break
        #After the for loop, remove from image black columns
        for i_col in range(offset-1):
            the_image[:, -i_col-1] = the_image[:, -offset]
        #Now fill the new image with new columns
        for i_col in range(cols_to_add):
            expanded_image[:, -i_col-1] = the_image[:, -offset]
        expanded_image[:, :num_cols] = the_image
    else: #Breast on the left
        side = "left"
        #Search for first column with non-zero values (some images have black column before breast tissue)
        offset = 0
        for i_col in range(num_cols):
            curr_col = the_image[:, i_col]
            if(curr_col.sum()/num_rows > 80):
                offset = i_col
                break
        #After the for loop, remove from image black columns
        for i_col in range(offset):
            the_image[:, i_col] = the_image[:, offset]
        #Now fill the new image with new columns
        for i_col in range(cols_to_add):
            expanded_image[:,i_col] = the_image[:, offset]
        expanded_image[:, cols_to_add:] = the_image
    return expanded_image.astype(np.uint8), cols_to_add, offset, side


def improve_resulting_segmentation(the_image, connectivity):

    """Function to post-process resulting image to remove very small regions and divide the ones that are separated by a small line.

    Parameter:
        image(numpy array): Image to work with.
        connectivity(int): Type of connectivity

    Returns:
        output_image(numpy array): Output image with corrected regions

   """

    for i in range(3):
        alt_image = np.zeros(the_image.shape)
        nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(the_image, connectivity)
        #First, separate regions that are connected by a thin line
        for i_label in range(nlabels-1):
            output_image = np.zeros(the_image.shape)
            output_image[labels==i_label+1] = 1 #Image with only the current region
            temp_image = cv.morphologyEx(output_image, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)))
            nlabels_temp, labels_temp, stats_temp, centroids_temp = cv.connectedComponentsWithStats(temp_image.astype(np.uint8), connectivity)
            if(nlabels_temp>2): #If region became two or more
                for i_label_temp in range(nlabels_temp-1):
                    alt_image[labels_temp==i_label_temp+1] = 1
            else:
                alt_image[labels==i_label+1] = 1
        #Second, discard very small regions (in case any)
        nlabels_n, labels_n, stats_n, centroids_n = cv.connectedComponentsWithStats(alt_image.astype(np.uint8), connectivity)
        output_image = np.zeros(the_image.shape)
        for i_label_n in range(nlabels_n-1):
            if(stats_n[i_label_n+1, 4]>= 50): #Area greater than 50 pixels
                output_image[labels_n==i_label_n+1] = 1
        the_image = output_image.astype(np.uint8)
    return output_image.astype(np.uint8)


def find_candidates(img, num_floors, debug = False):

    """Function to find masses candidates based on linear structuring elements and thresholding

    Parameter:
        img(numpy array): Full resolution original with original bit depth

    Returns:
        high_res(numpy array): Output binary image with the regions corresponding to the candidates

   """

    #Remove background from the image and apply CLAHE
    img = remove_background(bring_to_256_levels(img))
    if debug:
        show_image(img, "After removing background and CLAHE", 0.2)
        cv.waitKey(0)
    # Step 1: Gaussian pyramidal decomposition
    for i in range(num_floors):
        low_res = cv.pyrDown(img)
        img = low_res
    init_rows = low_res.shape[0]
    init_cols = low_res.shape[1]
    low_res_orig = low_res.copy()
    #Here we need an additional step to avoid linear SEs to give a wrong image
    #The proposed solution is to expand the image
    low_res, addedPixels, offset, side = expand_image(low_res)
    #Carefully chose the sizes for the linear structuring elements
    big_lines = np.arange(31, 241, 25, dtype = np.uint8)
    #Create output image
    output_image = np.zeros((init_rows, init_cols), dtype=np.uint8)
    for curr_big_line in big_lines:
        #Define d1 as 0.7*d2. This value was found to bring a good representation of the objects in terms of intensity
        small_lines = int(curr_big_line*(0.7))
        #Make d1 odd:
        if(small_lines%2 == 0):
            small_lines = small_lines+1
        #Define array for storing the output image for the current scale
        total = np.zeros(low_res.shape)
        #For each angle of the structuring element, apply tophat followed by opening
        for i in range(12):
            curr_big = cv.morphologyEx(low_res, cv.MORPH_TOPHAT, getLinearSE(curr_big_line, i+1))
            curr_small = cv.morphologyEx(curr_big, cv.MORPH_OPEN, getLinearSE(small_lines, i+1))
            total = total + cv.GaussianBlur(curr_small, (5,5), 0)
        #Recover image (remove expansion)
        if(side == "left"):
            total = total[:, addedPixels:]
            total[:,:offset] = np.zeros((init_rows, offset), dtype=np.uint8)
        else:
            total = total[:, :(total.shape[1] - addedPixels)]
            total[:, -offset-1:] = np.zeros((init_rows, offset), dtype=np.uint8)
        #Show image if needed
        if debug:
            show_image(bring_to_256_levels(total), "After linear SE", 0.7)
            cv.waitKey(0)
        #Normalize to 0-255
        total = bring_to_256_levels(total)
        #Compute Otsu's threshold
        threshold_value, _ = cv.threshold(total , 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        #Create 16 threshold values (from the otsu threshold to the maximum)
        steps = np.linspace(threshold_value, 254, 16)
        #Define image for temporary thresholded images and define connectivity
        base_image = np.zeros(total.shape)
        connectivity = 8
        #Apply thresholds
        for curr_thresh in steps:
            #Apply current threshold
            ret, thresholded_image = cv.threshold(total, int(curr_thresh), 255, cv.THRESH_BINARY)
            analysed_image = discard_regions_processing(thresholded_image, connectivity, small_lines, curr_big_line)
            base_image = base_image.astype(np.uint8) | analysed_image.astype(np.uint8)
        if debug:
            show_image(bring_to_256_levels(base_image), "After MLT")
            cv.waitKey(0)
        #Store results in same image using OR
        output_image = output_image | base_image
    endT = time.time()
    elapsed = endT - start
    print("Time elapsed for this image: ", elapsed)
    if debug:
        show_image(output_image, "Output image", 0.2)
        cv.waitKey(0)
    #Apply erosion first
    output_image = cv.morphologyEx(output_image, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)))
    if debug:
        show_image(bring_to_256_levels(output_image), "After Erosion")
        cv.waitKey(0)
    #Holes filling (As proposed in https://stackoverflow.com/questions/10316057/filling-holes-inside-a-binary-object)
    des = output_image.copy()
    contour = cv.findContours(des,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)[1]
    for cnt in contour:
        cv.drawContours(des,[cnt],0,255,-1)
    if debug:
        show_image(bring_to_256_levels(des), "After filling holes")
        cv.waitKey(0)
    #Remove regions that are too big or too small and analyse regions to divide the ones connected by thin line
    des_rem  = improve_resulting_segmentation(bring_to_256_levels(des), connectivity)
    des = des & des_rem.astype(np.uint8)
    #Bring result to full resolution and return it
    for i in range(num_floors):
        high_res = cv.pyrUp(des)
        des = high_res
    #high_res is what I need to return
    return high_res