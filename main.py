from feature_extraction.build_features_file import build_features_file, read_images
from main_flow import flow
from feature_extraction.training_data import prepate_datasets

path = "/home/jhonmgb/datasets/tmp/"
image = "24065584_d8205a09c8173f44_MG_L_CC_ANON.tif"

#[raw_im_Path, gt_im_path, raw_images, gt_images, false_positive_path, true_positive_path] = read_images(path)

#build_features_file(raw_images, raw_im_Path, gt_images, gt_im_path, false_positive_path, true_positive_path, path)

#flow.get_rois_from_image(path, image)

prepate_datasets(path)