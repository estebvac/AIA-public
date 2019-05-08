import cv2
import numpy as np
from . import util


def iris_filter(image, lines = 5, minDim = 10, maxDim = 15):
    """
    Applies the Iris filter over the given greyscale image

    Parameters
    ----------
    lines : int
        Angles to consider during the iris filter calculation
    minDim : int
        minimum radius of the iris filter
    maxDim : int
        maximum radius of the iris filter

    Returns
    -------
    np.ndarray
        a matrix with the iris filter response at every pixel
    """

    image = image.astype(np.float)
    kernely = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernelx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Calculate the X and Y gradients using Prewitt Masks
    img_prewittx = cv2.filter2D(image, -1, kernelx)
    img_prewitty = cv2.filter2D(image, -1, kernely)

    angles = np.ceil(360 / lines);

    out_filter = np.empty(shape=image.shape, dtype=np.float32)

    for i in range(0, img_prewitty.shape[0]):
        for j in range(0, img_prewitty.shape[1]):

            convergence_coeffs = np.empty(shape=(lines, 1), dtype=np.float32)
            current_line = 0
            if image[i,j] == 0:
                continue

            center = np.array([i, j])

            # Calculate the coefficient indexes per every line projected at a certain angle
            for angle in range(0, 360, angles.astype(np.uint)):
                midPoint = util.calculate_point(center, minDim, angle)
                finalPoint = util.calculate_point(center, maxDim, angle)
                directionVector = (center - finalPoint)
                directionVector[1] = -directionVector[1]
                iterator = util.create_line_iterator(midPoint, finalPoint, image.shape[0], image.shape[1])

                gradient = np.empty(shape=(iterator.shape[0], 2), dtype=np.float32)
                gradient[:, 0] = img_prewittx[iterator[:, 0].astype(np.uint), iterator[:, 1].astype(np.uint)]
                gradient[:, 1] = img_prewitty[iterator[:, 0].astype(np.uint), iterator[:, 1].astype(np.uint)]

                convergence_coeffs[current_line] = __calculate_coeff(gradient, directionVector)
                current_line += 1

            out_filter[i,j] = convergence_coeffs.sum()/lines;
    return out_filter


def __calculate_coeff(gradient, direction_vector):

    convergence_coeff = np.empty(shape=(gradient.shape[0], 1), dtype=np.float32)
    convergence_index = np.empty(shape=(gradient.shape[0], 1), dtype=np.float32)

    norm = np.dot(direction_vector, direction_vector)
    ## FIND THE CONVERGENCE INDEX OF A REGION R.
    for k in range(0, gradient.shape[0]):
        alpha = np.dot(gradient[k, :], direction_vector) / norm
        projection = alpha * direction_vector
        convergence_index[k] = \
            0 if (np.linalg.norm(projection) == 0)\
                else np.dot(gradient[k, :], projection)/(np.linalg.norm(gradient[k,:])*np.linalg.norm(projection))

    for i in range(0, convergence_index.shape[0]):
        convergence_coeff[i] = np.sum(convergence_index[:i])/(i+1)
    return 0 if convergence_coeff.shape[0] == 0 else convergence_coeff.max();

