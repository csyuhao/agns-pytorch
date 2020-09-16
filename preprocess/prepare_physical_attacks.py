import os
import cv2
import sys
import argparse
import numpy as np
from scipy.io import loadmat, savemat


def main(args):
    dataset_dir = args.input
    cropped_info = args.output
    eyeglasses_centers = args.eyeglass_centers
    output_size = args.output_size
    threshold_rgb = (70, 70, 70)

    centers = loadmat(eyeglasses_centers)[
        'eyeglass_marks_centers'].astype(np.float32)
    centers = np.array([[-1 + 2 * x / (output_size - 1.0), -1 + 2 *
                         y / (output_size - 1.0)] for (x, y) in centers], dtype=np.float32)

    filepaths = []
    for home, dirname, filenames in os.walk(dataset_dir):
        for filename in filenames:
            filepath = os.path.join(home, filename)
            filepaths.append(filepath)

    for filepath in filepaths:
        if filepath.endswith('.jpg') is False and filepath.endswith('.png') is False:
            continue
        aligned_img = cv2.imread(filepath)

        # threshold images
        img = None
        _, _, c = aligned_img.shape
        for idx in range(c):
            if idx % 2 == 0:
                binary_img = aligned_img[:, :, idx] < threshold_rgb[idx]
            else:
                binary_img = aligned_img[:, :, idx] > threshold_rgb[idx]
            img = (img & binary_img) if img is not None else binary_img
        img = (255.0 * img).astype(np.uint8)

        # calculating minimal rectangles
        def calculate_area(contour):
            area = cv2.contourArea(contour)
            return area

        locs = []
        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=calculate_area, reverse=True)
        for contour in contours[:7]:
            x, y, w, h = cv2.boundingRect(contour)
            locs += [(int(x + w / 2), int(y + h / 2))]

        # sorting locations
        def sorting_index_1(value):
            return value[0]

        def sorting_index_2(value):
            return value[1]

        locs.sort(key=sorting_index_2, reverse=False)
        tmp_1 = locs[:3].copy()
        tmp_1.sort(key=sorting_index_1, reverse=False)
        tmp_2 = locs[3:].copy()
        tmp_2.sort(key=sorting_index_1, reverse=False)
        locs = tmp_1 + tmp_2

        # transform parameters
        center_locs = np.array(locs, dtype=np.float32)
        center_locs = np.array([[-1 + 2 * x / (output_size - 1.0), -1 + 2 * y / (
            output_size - 1.0)] for (x, y) in center_locs], dtype=np.float32)
        matrix, _ = cv2.findHomography(center_locs, centers, method=cv2.RANSAC)
        filepath = filepath.replace(dataset_dir, cropped_info)
        sub_dirs, _ = os.path.split(filepath)
        if os.path.exists(sub_dirs) is False:
            os.makedirs(sub_dirs)
        savemat(filepath.replace('.jpg', '.mat').replace('.png', '.mat'), {
            'matrix': matrix
        })


def parse(argv):
    parser = argparse.ArgumentParser('Prepare Physical Attacker Parameters')
    parser.add_argument('--input', type=str, required=True,
                        help='file path of facebank')
    parser.add_argument('--output', type=str, required=True,
                        help='file path of cropped dataset')
    parser.add_argument('--output_size', type=int,
                        default=224, help='size of cropped images')
    parser.add_argument('--eyeglass_centers', type=str,
                        default=r'..\data\eyeglass_marks_centers.mat', help='eyeglasses centers coordinate')
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse(sys.argv[1:])
    main(args)
