""" Import all required libraries """
from evaluation.dice_similarity import *
from false_positive_reduction.false_positive_reduction import *
from feature_extraction.contour_features import calculate_contour_features
from feature_extraction.feature_extraction_optical_density import *
from candidates_detection import find_candidates
import pandas as pd


#  Read the dataset
raw_im_Path = r"C:\Users\esteb\Documents\AIA_PROJECT\dataset\images"
gt_im_Path = r"C:\Users\esteb\Documents\AIA_PROJECT\dataset\groundtruth"
raw_images = [f for f in listdir(raw_im_Path) if isfile(join(raw_im_Path, f))]
gt_images = [f for f in listdir(gt_im_Path) if isfile(join(gt_im_Path, f))]

# Paths to store the ROIs
false_positive_path = r"C:\Users\esteb\Documents\AIA_PROJECT\dataset\false_positive"
true_positive_path = r"C:\Users\esteb\Documents\AIA_PROJECT\dataset\true_positive"

# Ground truth images counter
gt_counter = 0

# create a Dataframe to store the features
d = []

img = cv2.imread(r"C:\Users\esteb\Documents\AIA_PROJECT\dataset\images\22427705_d713ef5849f98b6c_MG_L_CC_ANON.tif", cv2.IMREAD_UNCHANGED)
img = cv2.imread(r"C:\Users\esteb\Documents\AIA_PROJECT\dataset\images\22670147_e1f51192f7bf3f5f_MG_R_ML_ANON.tif", cv2.IMREAD_UNCHANGED)
img = cv2.imread(r"C:\Users\esteb\Documents\AIA_PROJECT\dataset\images\50994408_cc9e66c5b31baab8_MG_R_CC_ANON.tif", cv2.IMREAD_UNCHANGED)
img = cv2.imread(r"C:\Users\esteb\Documents\AIA_PROJECT\dataset\images\22427840_bbd6a3a35438c11b_MG_R_CC_ANON.tif", cv2.IMREAD_UNCHANGED)
img = cv2.imread(r"C:\Users\esteb\Documents\AIA_PROJECT\dataset\images\24065707_5291e1aee2bbf5df_MG_R_ML_ANON.tif", cv2.IMREAD_UNCHANGED)

all_scales = find_candidates(img, 3, debug=False)
all_scales = border_false_positive_reduction(all_scales, img)
for slice_counter in np.arange(all_scales.shape[2]):
    slice = all_scales[:, :, slice_counter]
    plt.imshow(slice, cmap='gray')
    plt.show()


# Loop over all the images
for img_counter in range(0, len(raw_images) - 2):

    # Process the original image
    img = cv2.imread(join(raw_im_Path, raw_images[img_counter]), cv2.IMREAD_UNCHANGED)
    all_scales = find_candidates(img, 3, debug=False)
    all_scales = border_false_positive_reduction(all_scales, img)

    if raw_images[img_counter] == gt_images[gt_counter]:

        # Dilate to remove holes in the GT
        gt = cv2.imread(join(gt_im_Path, gt_images[gt_counter]), 0)
        kernel = np.ones((51, 51), np.uint8)
        gt = cv2.dilate(gt, kernel, iterations=1)
        true_positives = True

        # Create markers of the ground truth
        markers_gt = np.zeros_like(img)
        _, contours_gt, _ = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for count in range(len(contours_gt)):
            cv2.drawContours(markers_gt, contours_gt, count, (count + 1), -1)

        # Create a GT image that matches with the input image size
        gt = np.zeros_like(img)
        cv2.drawContours(gt, contours_gt, -1, 1, -1)

        # Increase the ground truth counter
        if gt_counter < len(gt_images) - 1:
            gt_counter = gt_counter + 1

    else:
        gt = np.zeros_like(img)
        true_positives = False


    # Loop around the different segmentation sizes
    for slice_counter in np.arange(all_scales.shape[2]):
        slice = all_scales[:, :, slice_counter]
        if slice.max() > 0:
            slice = 255 * slice
            _, contours, _ = cv2.findContours(slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            matched_gt = slice * gt
            markers = np.zeros_like(img)
            if true_positives:
                if np.sum(matched_gt) > 0:
                    # Get the contours that contain TP ROIs
                    true_positives = True
                    for count in range(len(contours)):
                        cv2.drawContours(markers, contours, count, (count + 1), -1)
                    contain_tp = np.unique(markers * gt) - 1
                else:
                    true_positives = False

            for roi_counter in np.arange(len(contours)):

                # Prepare the image to store
                roi_img, boundaries = extract_ROI(contours[roi_counter], img)
                roi = np.copy(roi_img)

                bw_img = np.zeros_like(img)
                cv2.drawContours(bw_img, contours, roi_counter, 1, -1)
                roi_bw, _ = extract_ROI(contours[roi_counter], bw_img)

                roi_img = cv2.normalize(roi_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                roi_img = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
                _, contours_bw, _ = cv2.findContours(np.uint8(roi_bw * 255), cv2.RETR_TREE,
                                                     cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(roi_img, contours_bw, -1, (255, 0, 0), 3)

                # Name of the image to store
                path_name = [raw_images[img_counter][0:-4], "_layer_", str(slice_counter), "_roi_",
                             str(roi_counter), ".tif"]
                path_name = "".join(path_name)
                chdir(false_positive_path)
                label = 'FP'
                dice_index = 0
                jaccard = 0

                # Match with a  GT ROI:
                if np.sum(contain_tp == roi_counter):

                    # Extract the region of interest that matches the ground truth
                    roi_gt, _ = extract_ROI(contours[roi_counter], markers_gt)
                    if (np.sum(roi_gt)>0):
                        gt_label = np.unique(roi_gt)
                        if len(gt_label) == 1:
                            gt_label = np.int(gt_label - 1)
                        else:
                            gt_label = np.int(gt_label[1] - 1)

                        gt_img = np.zeros_like(img)
                        cv2.drawContours(gt_img, contours_gt, gt_label, 1, -1)

                        dice_index, jaccard = dice_similarity(bw_img, gt_img)

                        if jaccard > 0.2 and true_positives:
                            chdir(true_positive_path)
                            label = 'TP'

                # Contour features
                cnt_features = calculate_contour_features(contours[roi_counter])

                # Haralick Features
                textures = feature_extraction_haralick(roi)

                # Hu moments:
                hu_moments = feature_hu_moments(contours[roi_counter])

                # Multi-Scale Local Binary Pattern features:
                lbp = multi_scale_lbp_features(roi)

                # TAS features
                tas_features = feature_tas(roi_bw)

                d.append({
                    'File name': path_name,
                    'label': label,
                    'Image number': img_counter,
                    'Slice': slice_counter,
                    'Roi number': roi_counter,
                    'Boundaries': boundaries,
                    'Dice': np.array(dice_index),
                    'Jaccard': jaccard,
                    'Contour features': cnt_features,
                    'Haralick Features': textures,
                    'Hu moments': hu_moments,
                    'lbp': lbp,
                    'TAS features': tas_features
                })


                cv2.imwrite(path_name, roi_img)
                print("Writing: image N. ", img_counter, " slice: ", str(slice_counter), "of",
                      str(all_scales.shape[2] - 1), " ROI: ", str(roi_counter), " of ", str(len(contours) - 1),
                          " DICE: ", str(dice_index))

df = pd.DataFrame(d)
df.to_csv(r'C:\Users\esteb\Documents\AIA_PROJECT\dataset\features.csv', index=False)
