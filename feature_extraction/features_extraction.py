import cv2
import mahotas as mt
import numpy as np
from math import copysign, log10
from skimage.feature import hog

def feature_extraction_haralick_candidate(masked_roi):
    textures = mt.features.haralick(masked_roi, ignore_zeros = True)
    return textures.mean(axis=0)


def feature_extraction_haralick(roi):
    textures = mt.features.haralick(roi)
    return textures.mean(axis=0)


# Hu moments (Shape features)
def f_hu_moments(bin_roi):
    #_, bin_roi = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY)
    central_moments = cv2.moments(bin_roi, binaryImage = True)
    hu_moments = cv2.HuMoments(central_moments)
    # Log scale transform
    for i in range(0, 7):
        hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))
    return hu_moments


# Hu moments (Shape features)
def feature_hu_moments(contour):
    hu_moments = cv2.HuMoments(cv2.moments(contour))
    # Log scale transform
    for i in range(0, 7):
        hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))
    return hu_moments.reshape(-1)


def multi_scale_lbp_features(roi):
    roi_img = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    roi_or = np.copy(roi_img)
    r = 1
    R = 1
    i = 0
    lbp = np.zeros((5,36))
    while R < 35:
        lb_p = mt.features.lbp(roi_img, np.rint(R), 8, ignore_zeros=False)
        lb_p = lb_p / np.sum(lb_p)
        lbp[i, :] = lb_p
        r_1 = r
        r = r_1 * (2 / (1 - np.sin(np.pi / 8)) - 1)
        R = (r + r_1) / 2
        if np.floor(r - r_1) % 2 == 0:
            k_size = np.int(np.ceil(r - r_1))
        else:
            k_size = np.int(np.floor(r - r_1))
        std_dev = (k_size / 2)
        roi_img = cv2.GaussianBlur(roi_or, (k_size, k_size), std_dev)
        i += 1

    return lbp

def feature_tas(roi):
    roi_img = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return mt.features.tas(roi_img)

def features_hog(roi):
    width = np.int(roi.shape[0] / 10)
    height = np.int(roi.shape[1] / 10)
    w_t = np.int((roi.shape[0] - width * 10) / 2)
    h_t = np.int((roi.shape[1] - height * 10) / 2)
    crop_roi = roi[w_t: w_t + 10*width, h_t: h_t + 10*height]
    f_hog = hog(crop_roi, orientations=8, pixels_per_cell=(width, height),
             cells_per_block=(1, 1), visualize=False, multichannel=False)
    return f_hog


