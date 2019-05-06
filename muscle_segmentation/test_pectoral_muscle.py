""" Import all required libraries """

from os import listdir
from os.path import isfile, join
from muscle_segmentation import *

"""  Read the dataset """
# myPath = r"C:\Users\esteb\Documents\AIA_PROJECT\dataset\images"
raw_im_Path = R"C:\Users\esteb\Documents\PECTORAL MUSCLE\dataset\images"
gt_im_Path = R"C:\Users\esteb\Documents\PECTORAL MUSCLE\dataset\groundtruths"
mask_im_Path = R"C:\Users\esteb\Documents\PECTORAL MUSCLE\dataset\masks"


raw_images = [f for f in listdir(raw_im_Path) if isfile(join(raw_im_Path, f))]
gt_images = [f for f in listdir(gt_im_Path) if isfile(join(gt_im_Path, f))]
masks_images = [f for f in listdir(mask_im_Path) if isfile(join(mask_im_Path, f))]

performance = np.zeros(len(raw_images))

for n in range(0, len(raw_images)-2):
    img = cv2.imread(join(raw_im_Path, raw_images[n]), 0)
    gt_img = cv2.imread(join(gt_im_Path, gt_images[n]), 0)
    mask_img = cv2.imread(join(mask_im_Path, masks_images[n]), 0)
    img = np.asarray(img)
    gt_img = np.asarray(gt_img)
    mask_img = np.asarray(mask_img)
    segmented_breast = pectoral_muscle_segmentation(img, debug=False)

    # True Positives match
    true_positives = gt_img * segmented_breast
    true_positives = np.sum(1. * (true_positives > 0))

    # True negatives match
    true_negatives = (mask_img - gt_img) * (255 - segmented_breast)
    true_negatives = np.sum(1. * (true_negatives > 0))

    # Total number of samples within the breast
    total_samples = np.sum(1. * (mask_img > 0))

    # Performance measure
    perf = (true_negatives + true_positives) / total_samples
    performance[n] = min(1, perf)
    print("img ", str(n), ":  ", str(performance[n]))

    # Check low performance samples
    if perf < 0.90:
        plt.subplot(1, 2, 1)
        plt.imshow(segmented_breast)
        plt.subplot(1, 2, 2)
        plt.imshow(gt_img)
        plt.show()
        _, segmented_breast = pectoral_muscle_segmentation(img, debug=False)
print("Average: ", str(performance.mean()))