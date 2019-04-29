import cv2
import numpy as np

def remove_background(image):

	"""Function to remove the background of the image.

    Parameter:
    	image(numpy array): Original image.

    Return:
    	img_new(numpy array): Image without a background.

   """

    # OTSU's thresholding
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    max_cont = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [max_cont], -1, 255, -1)
    bit_and = cv2.bitwise_and(image, image, mask=mask)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_new = clahe.apply(bit_and)

    return img_new
