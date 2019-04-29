import numpy as np

def is_right(image):

    left = np.mean(np.array_split(np.sum(image, axis=0), 2)[0])
    right = np.mean(np.array_split(np.sum(image, axis=0), 2)[1])

    if right > left:
        return True
    else:
        return False
