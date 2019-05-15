import cv2
import mahotas as mt
import numpy
from math import copysign, log10
from skimage.transform import integral_image
from skimage.feature import haar_like_feature


def feature_extraction_haralick(roi):
    textures = mt.features.haralick(roi)
    return textures.mean(axis=0)


# Hu moments (Shape features)
def f_hu_moments(roi):
    _, bin_roi = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY)
    central_moments = cv2.moments(bin_roi)
    hu_moments = cv2.HuMoments(central_moments)
    # Log scale transform
    for i in range(0, 7):
        hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))
    return hu_moments


# Haar-like features
def feature_haar(roi, feature_coord=None):
    feature_type = 'type-2-x', 'type-2-y', 'type-4'
    ii = integral_image(roi)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1], feature_type=feature_type, feature_coord=feature_coord)



image = cv2.imread('22614266_1e5c3af078f74b05_MG_L_ML_ANON_layer_2_roi_2.tif')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

features = feature_extraction_haralick(gray)
print('Haralick features: ',features.shape)

HM = f_hu_moments(gray)
print('Hu moments: ', HM.shape)

features = numpy.append(features, HM)
print('Features: ', features.shape)

#gray = cv2.resize(gray,(19,19))
haar_feat = feature_haar(gray)
print('Haar features: ', haar_feat.shape)