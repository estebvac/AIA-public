import numpy as np

def is_right(image):

	"""Function to define the breast orientation.

    Parameter:
    	image(numpy array): Image to work with.

    Returns:
    	True(boolean): If the orientation is right.
    	False(boolean): If the orientation is left.

   """

    left = np.mean(np.array_split(np.sum(image, axis=0), 2)[0])
    right = np.mean(np.array_split(np.sum(image, axis=0), 2)[1])

    if right > left:
        return True
    else:
        return False
