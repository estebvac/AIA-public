from feature_extraction.build_features_file import build_features_file, read_images

path = r"C:\AIA_PROJECT\dataset\\"

[raw_im_Path, gt_im_path, raw_images, gt_images, false_positive_path, true_positive_path] = read_images(path)

build_features_file(raw_images, raw_im_Path, gt_images, gt_im_path, false_positive_path, true_positive_path, path)

