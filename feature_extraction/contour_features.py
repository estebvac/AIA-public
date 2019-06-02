import cv2
import numpy as np


def calculate_contour_features(contour):
    list = [None] * 8
    list[0] = solidity(contour)
    list[1] = convex_area(contour)
    list[2] = rectangularity(contour)
    list[3] = eccentricity(contour)
    list[4] = center_of_gravity(contour)
    list[5] = circularity_ratio(contour)
    list[6] = min_max_axis_length(contour)
    list[7] = ellipse_variance(contour)

    return list


def solidity(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return float(area)/hull_area

def convex_area(contour):
    convex_hull = cv2.convexHull(contour)
    return cv2.arcLength(convex_hull, True)/cv2.arcLength(contour, True)

def rectangularity(contour):
    rectangle = cv2.minAreaRect(contour)
    rectangle_points = cv2.boxPoints(rectangle)
    return cv2.contourArea(contour)/cv2.contourArea(rectangle_points)

def eccentricity(contour):
    rectangle = cv2.minAreaRect(contour)
    width, height = rectangle[1]
    return height/width


def center_of_gravity(contour):
    moments = cv2.moments(contour)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return np.asarray([cx, cy])

def circularity_ratio(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return area/(perimeter * perimeter)

def min_max_axis_length(contour):
    rectangle = cv2.minAreaRect(contour)
    width, height = rectangle[1]
    return np.sort(np.asarray([width, height]))

def ellipse_variance(contour):
    ellipse = cv2.fitEllipse(contour)
    poly =\
        cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
    poly = poly.astype('float64')
    covar, _ = cv2.calcCovarMatrix(poly, 0, flags=cv2.COVAR_NORMAL + cv2.COVAR_ROWS)
    inv_covar = np.linalg.inv(covar + np.finfo(float).eps)
    gravity = center_of_gravity(contour);
    di = np.zeros((1, poly.shape[0]))
    i = 0
    mr = 0
    for point in poly:
        vt = point - gravity
        v = np.transpose(vt)
        d = np.sqrt(np.matmul(np.matmul(vt, inv_covar), v))
        di[0, i] = d
        mr += d
        i += 1

    mr = mr/i

    di = di - mr
    di = di * di
    dr = np.sqrt(np.sum(di)/i)
    return dr