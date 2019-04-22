import cv2
import numpy as np
from matplotlib import pyplot as plt


def normalize_image(x_input):
    """
    Function to normalize the image limits between 0 and 1

    :param x_input: Input image to normalize
    :return:        Image with adjusted limits between [0., 1.]
    """
    n = x_input - x_input.min()
    n = n / x_input.max()
    return n


def normalize_and_equalize(x_input):
    """
    Equalizes the histogram of a grayscale of any type

    :param x_input: Input image to equalize
    :return:        Equalized image with limits between [0, 255]
    """
    out = 255 * normalize_image(x_input)
    out = cv2.equalizeHist(out.astype(np.uint8))
    return out


def make_convex(x_input):
    """
    Returns the convex hull of the biggest blob in x_input.

    :param x_input: Binary image containing the blob to make convex
    :return:        Convex hull of the biggest blob of the image
    """
    # Make the biggest contour convex
    im2, contours, hierarchy = \
        cv2.findContours(x_input, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = []

    # calculate points for each contour
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))

    # create an empty black image
    out = np.zeros_like(x_input)
    cv2.drawContours(out, hull, -1, 255, -1)
    return out


def auto_canny(image, sigma=0.33):
    """
    Automatic Canny implementation, taken from:
    https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-
    canny-edge-detection-with-python-and-opencv/

    :param image:   Input image
    :param sigma:   The desired deviation of the Canny threshold
    :return:        The edges of the image
    """

    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    upper = int(min(255, (1.0 - sigma) * v))
    edged = cv2.Canny(image, 5, upper)

    # return the edged image
    return edged


def remove_background(x_input):
    """
    Removes the pixels of the mammography that does not belong to the breast

    :param x_input: Input image
    :return:        Breast mask (binary)
    """
    # Normalise the image
    x_normalized = 255 * normalize_image(x_input)
    x_normalized = x_normalized.astype(np.uint8)

    # Threshold the image
    _, breast_bw = cv2.threshold(x_normalized,
                                 x_normalized.min(), x_normalized.max(),
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return breast_bw


def flip_and_crop(x_input, breast_bw):
    """
    Detect if the mammography is right or left and flips it if necessary, it creates a
    bounding rectangle around the breast pixels.

    :param x_input:     Input image
    :param breast_bw:   Mask of the breast pixels
    :return:            The following array:
                        breast_left:        breast left-aligned
                        breast_bw_left:     mask of the breast left-aligned
                        breast_orientation: string ("left/right")
                        boundaries:         coordinates of the bounding rectangle
    """
    # Get the dimension of the image
    inp_h, inp_w = x_input.shape

    # Define the boundaries
    x_b, y_b, w_b, h_b = cv2.boundingRect(breast_bw)
    boundaries = [x_b, y_b, w_b, h_b]

    # Crop the image
    breast_bound = x_input[y_b:y_b + h_b, x_b:x_b + w_b]
    breast_bw_left = breast_bw[y_b:y_b + h_b, x_b:x_b + w_b]
    breast_left = np.multiply(breast_bound, normalize_image(breast_bw_left))

    # Look if mammography is left or right
    if (x_b + w_b) > (inp_w - 10):
        breast_orientation = "right"
        breast_left = cv2.flip(breast_left, 1)
        breast_bw_left = cv2.flip(breast_bw_left, 1)
    else:
        breast_orientation = "left"

    return [breast_left, breast_bw_left, breast_orientation, boundaries]


def remove_muscle(x_input, x_bw, debug=False):
    """
    Removes the breast muscle of the image, and returns a image containing the segmented
    breast region and its  binary mask

    :param x_input:     Input image
    :param x_bw:        Mask of the breast pixels
    :param debug:       Boolean to show the intermediate process images
    :return:            A mask containing the breast without pectoral muscle and its mask

    """
    # Get the limits of the top and the bottom rows of the image:
    top_nonzero = np.asarray(np.nonzero(x_input[1, :]))
    bottom_nonzero = np.asarray(np.nonzero(x_input[-2, :]))
    top_len = top_nonzero.shape[1]
    bottom_len = bottom_nonzero.shape[1]

    # Prepare the output images
    breast_wo_muscle_bw = x_bw
    breast_wo_muscle = np.copy(x_input)

    # Check if it exists an inclination:
    if top_len > 2 * bottom_len or top_len > 50:

        # Get the coordinates of the region of interest
        x_input = x_input.astype(np.uint8)
        y_roi = np.int(0.8 * x_input.shape[0])
        x_roi = top_nonzero[0, -1] - 20

        if x_roi > 0:

            # Crop the roi of the image
            roi_muscle = x_input[0:y_roi, 0:x_roi]

            # Apply CLAHE in a 4x4 region with a clip limit of 2
            hand_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            img_clahe = hand_clahe.apply(roi_muscle)
            roi_clahe = normalize_and_equalize(img_clahe)

            # Apply Gaussian  blur to the image
            roi_smooth = cv2.GaussianBlur(roi_clahe, (7, 7), 71, cv2.BORDER_REPLICATE)

            # Remove the lower triangular part
            triangle_cnt = np.array([(0, 0),
                                     (0, np.int(y_roi * .9)),
                                     (np.int(x_roi), 0)])
            upper_triangle = np.zeros_like(roi_muscle).astype(np.uint8)
            cv2.drawContours(upper_triangle, [triangle_cnt], 0, 1, -1)
            top_regions = roi_smooth * upper_triangle

            # Apply K-means with K = 2 ( This approach obtained better results than OTSU)
            flat_roi = top_regions.reshape((-1, 1))
            flat_roi = np.float32(flat_roi)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, label, center = \
                cv2.kmeans(flat_roi, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # Convert back into uint8, and make original image
            center = np.uint8(center)
            center = np.uint8(center / np.max(center))
            res = center[label.flatten()]
            roi_clustered = res.reshape(roi_smooth.shape)

            # Extract the Connected Components of the image
            n_labels, labels, stats, centroids = \
                cv2.connectedComponentsWithStats(roi_clustered, 8)

            # Searching the muscle candidates
            muscle_cluster = np.zeros_like(roi_clustered)
            muscle_region_found = False

            for label in range(n_labels):
                if stats[label, 0] < 0.2 * muscle_cluster.shape[1] and \
                        stats[label, 1] < 0.2 * muscle_cluster.shape[0] and \
                        stats[label, 3] != muscle_cluster.shape[0]:
                    muscle_region_found = True
                    muscle_cluster[labels == label] = 1
            muscle_cluster = make_convex(muscle_cluster)

            if muscle_region_found:

                # Take last non-zero elements of the muscle candidates
                last_non_zero = muscle_cluster.shape[1] - \
                                (muscle_cluster != 0)[:, ::-1].argmax(1) - 1
                last_non_zero[last_non_zero == muscle_cluster.shape[1] - 1] = 0
                last_non_zero = last_non_zero.reshape((-1, 1))

                # Create coordinate vs last non-zero pairs and remove zeros
                x_position = np.arange(muscle_cluster.shape[0])
                x_position = x_position.reshape(-1, 1)
                points = np.concatenate((x_position,
                                         last_non_zero.reshape((-1, 1))), axis=1)
                points = points[np.all(points != 0, axis=1)]

                # Variable to store the segmented muscle space
                segmented_muscle = np.zeros_like(muscle_cluster).astype(np.uint8)

                if points[:, 0].size != 0:
                    # Fit a curve to the obtained points
                    order = 2
                    polynomial = np.poly1d(np.polyfit(points[:, 0], points[:, 1], order))
                    x_fit = np.arange(2 * np.max(points[:, 0]).astype(np.int))
                    x_fit = np.round(x_fit.reshape(-1, 1)).astype(np.int)
                    fitted = polynomial(x_fit).astype(np.int)
                    fitted = fitted.reshape(-1, 1)
                    breast_points = np.concatenate((fitted, x_fit), axis=1)

                    # Draw the contours of the segmented muscle
                    corner_points = np.array([[0, np.max(x_fit[:, 0])], [0, 0]])
                    breast_points = \
                        np.concatenate((corner_points, breast_points), axis=0)
                    cv2.drawContours(segmented_muscle, [breast_points], 0, 1, -1)

                    # Save the segmentation in the outputs arrays
                    segmented_muscle = 1 - segmented_muscle
                    breast_wo_muscle_bw[0:y_roi, 0:x_roi] = 255 * segmented_muscle
                    breast_wo_muscle = \
                        breast_wo_muscle * normalize_image(breast_wo_muscle_bw)

            if debug:
                plt.subplot(1, 3, 1)
                plt.imshow(roi_muscle)
                plt.title("ROI muscle")
                plt.subplot(1, 3, 2)
                plt.imshow(roi_clustered)
                plt.title("Connected components")
                plt.subplot(1, 3, 3)
                plt.imshow(breast_wo_muscle)
                plt.title("Segmented image")
                plt.show()

    return [breast_wo_muscle, breast_wo_muscle_bw]


def pectoral_muscle_segmentation(x_input, debug=False):
    """

    :param x_input: Input image to remove the breast muscle
    :param debug:   Bool to show the intermediate images
    :return:        A mask containing the breast mask without pectoral muscle
    """
    # resize image to speed up the process
    original_size = x_input.shape
    scale_percent = 10  # percent of original size
    width = int(x_input.shape[1] * scale_percent / 100)
    height = int(x_input.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(x_input, dim, interpolation=cv2.INTER_AREA)

    # Remove the background of the image
    breast_bw = remove_background(resized_image)
    [breast_left, breast_bw_left, breast_orientation, boundaries] = \
        flip_and_crop(resized_image, breast_bw)
    [x_b, y_b, w_b, h_b] = boundaries

    # Segment the Pectoral muscle
    [_, breast_wo_muscle_bw] = \
        remove_muscle(breast_left, breast_bw_left, debug)

    # Correct the orientation if needed
    if breast_orientation == "right":
        breast_wo_muscle_bw = cv2.flip(breast_wo_muscle_bw, 1)

    # Place back the segmented regions
    breast_bw[y_b:y_b + h_b, x_b:x_b + w_b] = breast_wo_muscle_bw

    # Restore to the original size
    breast_wo_muscle_bw = \
        cv2.resize(breast_bw, original_size[::-1], interpolation=cv2.INTER_NEAREST)
    breast_wo_muscle = x_input * normalize_image(breast_wo_muscle_bw)

    if debug:
        plt.subplot(1, 2, 1)
        plt.imshow(x_input)
        plt.title("Original image")
        plt.subplot(1, 2, 2)
        plt.imshow(breast_wo_muscle_bw)
        plt.title("Segmented region")
        plt.show()

    return [breast_wo_muscle, breast_wo_muscle_bw]
