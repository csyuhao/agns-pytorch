import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append('..')
from module.mtcnn.mtcnn import MTCNN


def collate_pil(x):
    out_x, out_y = [], []
    for xx, yy in x:
        out_x.append(xx)
        out_y.append(yy)
    return out_x, out_y


def main(args):
    batch_size = 1
    workers = 0 if os.name == 'nt' else 8
    dataset_dir = args.input
    cropped_dataset = args.output

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(
        image_size=args.output_size, margin=args.margin, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    dataset = datasets.ImageFolder(
        dataset_dir, transform=transforms.Resize((512, 512)))
    dataset.samples = [
        (p, p.replace(dataset_dir, cropped_dataset))
        for p, _ in dataset.samples
    ]
    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=collate_pil
    )

    for i, (x, y) in enumerate(loader):
        x = mtcnn(x, save_path=y)


def parse(argv):
    parser = argparse.ArgumentParser('Align Images')
    parser.add_argument('--input', type=str, required=True, help='file path of facebank')
    parser.add_argument('--output', type=str, required=True, help='file path of cropped dataset')
    parser.add_argument('--output_size', type=int, default=224, help='size of cropped images')
    parser.add_argument('--margin', type=int, default=0, help='marigin of cropped images')
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse(sys.argv[1:])
    main(args)
