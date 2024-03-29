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

    # Jiang_Cheng_Pu
    default_threshold_rgb = (120, 120, 120)

    threshold_rgbs = {
        "Gao_Zhen_Bo": (140, 130, 130),
        "Wang_Ming_Hui": (115, 115, 115),
    }

    centers = loadmat(eyeglasses_centers)[
        'eyeglass_marks_centers'].astype(np.float32)
    centers = np.array([[-1 + 2 * x / (output_size - 1.0), -1 + 2 *
                         y / (output_size - 1.0)] for (x, y) in centers], dtype=np.float32)

    filepaths = {}
    for dirname in os.listdir(dataset_dir):
        dirpath = os.path.join(dataset_dir, dirname)
        filepaths[dirname] = []
        for filename in os.listdir(dirpath):
            filepath = os.path.join(dirpath, filename)
            filepaths[dirname].append(filepath)

    for dirname in filepaths:
        for filepath in filepaths[dirname]:
            if dirname in list(threshold_rgbs.keys()):
                threshold_rgb = threshold_rgbs[dirname]
            else:
                threshold_rgb = default_threshold_rgb
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

            # print(dirname)
            if dirname == "Jiang_Cheng_Pu":
                img = (aligned_img[:, :, 1] - 10 > aligned_img[:, :, 0]) & (aligned_img[:, :, 1] - 10 > aligned_img[:, :, 2]) & (aligned_img[:, :, 1] > 100)
            elif dirname == "Liu_Hui_Sen":
                img = (aligned_img[:, :, 1] - 10 > aligned_img[:, :, 0]) & (aligned_img[:, :, 1] - 10 > aligned_img[:, :, 2]) & (aligned_img[:, :, 1] > 100)
            elif dirname == "Sun_Run_Geng":
                img = (aligned_img[:, :, 1] - 10 > aligned_img[:, :, 0]) & (aligned_img[:, :, 1] - 10 > aligned_img[:, :, 2]) & (aligned_img[:, :, 1] > 100)
            elif dirname == "Sun_Tian_Yi":
                img = (aligned_img[:, :, 1] - 10 > aligned_img[:, :, 0]) & (aligned_img[:, :, 1] - 10 > aligned_img[:, :, 2]) & (aligned_img[:, :, 1] > 100)
            elif dirname == "Tang_Xiang_Yun":
                img = (aligned_img[:, :, 1] - 10 > aligned_img[:, :, 0]) & (aligned_img[:, :, 1] - 10 > aligned_img[:, :, 2]) & (aligned_img[:, :, 1] > 100)
            elif dirname == 'Wang_Huan':
                img = (aligned_img[:, :, 1] >= aligned_img[:, :, 0]) & (aligned_img[:, :, 1] >= aligned_img[:, :, 2]) & (aligned_img[:, :, 1] > 140)
            elif dirname == 'Yu_Hao':
                img = (aligned_img[:, :, 1] - 10 >= aligned_img[:, :, 0]) & (aligned_img[:, :, 1] - 10 >= aligned_img[:, :, 2]) & (aligned_img[:, :, 1] > 120)
            elif dirname == 'Zhang_Jie':
                img = (aligned_img[:, :, 1] - 10 >= aligned_img[:, :, 0]) & (aligned_img[:, :, 1] - 10 >= aligned_img[:, :, 2]) & (aligned_img[:, :, 1] > 120)
            img = (255.0 * img).astype(np.uint8)

            # cv2.imshow('demo', img)
            # cv2.waitKey(0)

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
