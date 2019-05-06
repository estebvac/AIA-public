import cv2
import numpy as np


def sech_template_matching(image: np.ndarray) -> tuple:
    """Looks for the occurrence of the SECH templates (33x33 and 65x65) into the input image.

    Parameters
    ----------
    image : np.ndarray
        The numpy array representing the grayscale image

    Returns
    -------
    tuple
        A tuple where the first element is the result of applying the 33x33 SECH template,
        and the second one is the result of applying the 65x65 template

    """
    template33 = __create_sech_template()
    template33 =\
        cv2.normalize(template33, template33, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U);
    template65 = __create_sech_template(size = 65)
    template65 =\
        cv2.normalize(template65, template65, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U);

    return[ cv2.matchTemplate(image, template33, cv2.TM_CCOEFF),\
            cv2.matchTemplate(image, template65, cv2.TM_CCOEFF)]


def __create_sech_template(size: int = 33, beta: float = 0.08) -> np.ndarray:
    """Private function. Creates a custom SECH template.

    Parameters
    ----------
    size : int
        The size of the template. 33 by default

    beta : float
        Parametrizes the power of the SECH function (Values between 0 and 1). 0.08 by default

    Returns
    -------
    np.ndarray
        Numpy array with the generated templat

    """

    template = np.zeros([size, size], dtype=float)
    half = np.ceil(size /2);
    x = np.tile(np.arange(-half, half-1 if size%2 != 0 else half), (size,1))
    y = x.transpose()

    template =\
        2/(np.exp(beta * np.sqrt(x*x + y*y)) + np.exp(-beta * np.sqrt(x*x + y*y)))

    return template
