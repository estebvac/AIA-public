# Imports
import numpy as np
import os
import cv2 as cv
from preprocessing.is_right import is_right
from preprocessing.remove_background import remove_background
from preprocessing.muscle_segmentation import remove_background as get_breast_mask
from preprocessing.muscle_segmentation import normalize_image


def show_image(img_to_show, img_name, factor=1.0):
    """Function to display an image with a specific window name and specific resize factor

    Parameters
    ----------
    img_to_show : numpy array
        Image to be displayed
    img_name : string
        Name for the window
    factor : float
        Resize factor (between 0 and 1)

    """

    img_to_show = cv.resize(img_to_show, None, fx=factor, fy=factor)
    cv.imshow(img_name, img_to_show)


def bring_to_256_levels(the_image):
    """Function normalize image to 0-255 range (8 bits)

    Parameters
    ----------
    image : numpy array
        Original image.

    Returns
    -------
    img_new : numpy array
        Normalized image
    """

    if(the_image.max() == the_image.min()):
        return the_image.astype(np.uint8)
    img_as_double = the_image.astype(float)
    normalized = np.divide((img_as_double - np.amin(img_as_double)), (np.amax(img_as_double) - np.amin(img_as_double)))
    normalized = normalized*(pow(2, 8) - 1)
    return normalized.astype(np.uint8)


def getLinearSE(size, angle):
    """Function to create a linear SE with a specific size and angle.

    Parameters
    ----------
    image : numpy array
        Original image.
    size: int
        Size of the square that contains the linear SE
    angle : int
        Number that identifies an angle for the linear SE according to following options
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

    Returns
    -------
    SE_diff : numpy array
        Binary array of size <size> that contains linear SE with approximate angle given by <angle>
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


    Parameters
    ----------
    image : numpy array
        Original binary image.
    connectivity : int
        Connectivity for the region analysis
    d1 : int
        Minimum diameter for the regions
    d2 : int
        Maximum diameter for the regions


    Returns
    -------
    output_image : numpy array
        Image without regions that have an area that lies outside the mentioned range.
    """

    correction_factor = 1.3
    output_image = np.zeros(the_image.shape)
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(the_image, connectivity)
    area_range = np.array([(np.pi/4)*pow(d1,2), (np.pi/4)*pow(d2,2)])
    for i_label in range(nlabels):
        if(stats[i_label, 4]>= area_range[0] and stats[i_label, 4]<= area_range[1]):
            if(stats[i_label, 2] >= correction_factor*d2 or stats[i_label, 3] >= correction_factor*d2):
                continue
            else:
                output_image[labels==i_label] = 1
    return output_image


def improve_resulting_segmentation(the_image, connectivity):
    """Function to post-process resulting image to remove very small regions and divide the ones that are separated by a small line.


    Parameters
    ----------
    image : numpy array
        Image to work with.
    connectivity : int
        Connectivity for the region analysis

    Returns
    -------
    output_image : numpy array
        Output image with corrected regions
    """

    #output_image = np.zeros(the_image.shape)
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


def fill_holes(the_image):
    """Function to fill the holes in a binary image (from https://stackoverflow.com/questions/10316057/filling-holes-inside-a-binary-object)

    Parameters
    ----------
    the_image : numpy array
        Binary image to fill holes

    Returns
    -------
    des
        Binary image with holes filled
    """

    des = the_image.copy() #cv.bitwise_not(output_image)
    contour = cv.findContours(des,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)[1]
    for cnt in contour:
        cv.drawContours(des,[cnt],0,255,-1)
    return des


def find_candidates(img, num_floors, debug=False):
    """Function to find masses candidates based on linear structuring elements and thresholding

    Parameters
    ----------
    img : numpy array
        Full resolution original with original bit depth.
    num_floors : int
        Number of floors for the gaussian pyramid. 3 is the recommended value
    debug: bool
        Boolean to activate debugging mode

    Returns
    -------
    all_scales : numpy array (3D)
        Output binary images with the regions corresponding to the candidates. Last dimension indicates scale
    """

    #Remove background from the image and apply CLAHE
    filtered_image = remove_background(bring_to_256_levels(img)) #This returns the filtered image

    #Get side of the breast in the image
    if(is_right(filtered_image)):
        side = "r"
    else:
        side = "l"

    if debug:
        scaling_factor = 0.95
        show_image(filtered_image, "After removing background and CLAHE", 0.2)
        cv.waitKey(0)

    full_res_rows = filtered_image.shape[0]
    full_res_cols = filtered_image.shape[1]

    # Step 1: Gaussian pyramidal decomposition
    for i in range(num_floors):
        low_res = cv.pyrDown(filtered_image)
        filtered_image = low_res

    #Get mask of the breast
    img_bin = get_breast_mask(bring_to_256_levels(low_res)) #This returns a mask of the breast
    if debug:
        show_image(img_bin, "Mask", scaling_factor)
        cv.waitKey(0)

    orig_rows = low_res.shape[0]
    orig_cols = low_res.shape[1]
    #Crop image so that only breast region is analyzed
    x_b, y_b, w_b, h_b = cv.boundingRect(img_bin)
    boundaries = [x_b, y_b, w_b, h_b]
    # Crop the image
    breast_region = low_res[y_b:y_b + h_b, x_b:x_b + w_b]
    to_add = 300 #Number of pixels to add to the image

    if debug:
        show_image(breast_region, "Cropped image", scaling_factor)
        cv.waitKey(0)

    #Here we need an additional step to avoid linear SEs to give a wrong image
    #The proposed solution is to expand the image
    if(side == "l"):
        left_d = to_add
        right_d = 0
    else:
        left_d = 0
        right_d = to_add

    expanded_image = cv.copyMakeBorder(breast_region, 0, 0, left_d, right_d, cv.BORDER_REPLICATE)
    if debug:
        show_image(expanded_image, "Expanded image", scaling_factor)
        cv.waitKey(0)

    #Carefully chose the sizes for the linear structuring elements
    d1 = int(62/pow(2, num_floors-1))
    if d1%2 ==0 :
        d1 = d1+1
    d2 = int(482/pow(2, num_floors-1))
    if d2%2 == 0:
        d2 = d2+1

    #Generate sizes for d2
    big_lines = np.arange(d1, d2, 15, dtype = np.uint8)

    #Create output image
    output_image = np.zeros((orig_rows, orig_cols,1), dtype=np.uint8)

    for curr_big_line in big_lines:

        #Define d1 as 0.6*d2 (Several values were tested and this one provided a better representation)
        small_lines = int(curr_big_line*(0.6))

        #Make d1 odd:
        if(small_lines%2 == 0):
            small_lines = small_lines+1

        #Define array for storing the output image for the current scale
        total = np.zeros(expanded_image.shape)

        #For each angle of the structuring element, apply tophat followed by opening
        for i in range(12):
            curr_big = cv.morphologyEx(expanded_image, cv.MORPH_TOPHAT, getLinearSE(curr_big_line, i+1))
            curr_small = cv.morphologyEx(curr_big, cv.MORPH_OPEN, getLinearSE(small_lines, i+1))
            total = total + cv.GaussianBlur(curr_small, (5,5), 0)

        #Recover image (remove expansion)
        if(side == "l"):
            total = total[:, to_add:]
        else:
            total = total[:, :(total.shape[1] - to_add)]

        #Show image if needed
        if debug:
            show_image(bring_to_256_levels(total), "After linear SE", scaling_factor)
            cv.waitKey(0)

        #Normalize to 0-255
        total = bring_to_256_levels(total)

        #Compute Otsu's threshold
        threshold_value, _ = cv.threshold(total , 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        #Create 16 threshold values (from the otsu threshold to the maximum)
        steps = np.linspace(threshold_value, 254, 16)

        #Define image for temporary thresholded images and define connectivity
        region_with_candidates = np.zeros(total.shape)
        connectivity = 8
        #Apply thresholds
        for curr_thresh in steps:
            #Apply current threshold
            ret, thresholded_image = cv.threshold(total, int(curr_thresh), 255, cv.THRESH_BINARY)
            thresholded_image = fill_holes(thresholded_image)
            analysed_image = discard_regions_processing(thresholded_image, connectivity, small_lines, curr_big_line)
            region_with_candidates = region_with_candidates.astype(np.uint8) | analysed_image.astype(np.uint8)

        if debug:
            show_image(bring_to_256_levels(region_with_candidates), "After MLT", scaling_factor)
            cv.waitKey(0)

        #Reconstruct image with the size of the original one (before cropping)
        complete_image_curr_scale = np.zeros_like(low_res, dtype = np.uint8) #((orig_rows, orig_cols), dtype = np.uint8)
        complete_image_curr_scale[y_b:y_b + h_b, x_b:x_b + w_b] = region_with_candidates
        if debug:
            show_image(bring_to_256_levels(complete_image_curr_scale), "Current scale candidates", scaling_factor)

        #Store results 3D image
        if(complete_image_curr_scale.min() != complete_image_curr_scale.max()):
            output_image = np.dstack((output_image, complete_image_curr_scale.astype(np.uint8)))

    #Remove first element, which contains only zeros
    output_image = output_image[:,:,1:]

    if debug:
        for i in range(output_image.shape[2]):
            show_image(bring_to_256_levels(output_image[:,:,i]), "Output image at scale " + str(i), scaling_factor)
            cv.waitKey(0)

    #Build full resolution images
    all_scales = np.zeros((full_res_rows, full_res_cols, output_image.shape[2]), dtype = np.uint8)
    for i in range(all_scales.shape[2]):
        curr_image = output_image[:,:,i]
        all_scales[:,:,i] = cv.resize(curr_image, (full_res_cols, full_res_rows))

    #all_scales is what I need to return
    return all_scales
