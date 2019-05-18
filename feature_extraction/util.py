import cv2
import numpy as np;


def create_line_iterator(P1, P2, width, height):
    """
    DISCLAIMER: This line iterator code has been obtained and modified from:
    https://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator

    Produces and array that consists of the coordinates of each pixel in a line between two points

    Parameters
    ----------
    P1 : np.ndarray
        coordinate of the first point (x,y)
    P2 : np.ndarray
        coordinate of the second point (x,y)
    width : int
        the width of the reference matrix where the iterator will be used
    height : int
        the height of the reference matrix where the iterator will be used

    Returns
    -------
    np.ndarray
        the coordinates of each pixel in the radii (shape: [numPixels, 2], row = [x,y])
    """
    # define local variables for readability
    imageH = height
    imageW = width
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 2), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    return itbuffer


def calculate_point(center, rad, angle):
    """
    Calculate the coordinates from an central point at a radius rad, over an angle.

    Parameters
    ----------
    center : np.ndarray
        coordinate of the first point (x,y)
    rad : np.ndarray
        radius
    angle : int
        angle

    Returns
    -------
    np.ndarray
        the final point at the radius/angle from the central point
    """
    angle = 2*np.pi - np.radians(angle)
    out = np.array([0, 0])
    out[0] = np.int_(center[0] + rad * np.cos(angle))
    out[1] = np.int_(center[1] + rad * np.sin(angle))
    return out

def extract_contours(mask):
    _, _, a = mask.shape
    final_contours = [None] * a
    for i in range(0, a):
        img = mask[:,:,i].copy()
        _, contours, _ = cv2.findContours(img,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        final_contours[i] = contours

    return np.asarray(final_contours)

