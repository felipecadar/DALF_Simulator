import cv2
import numpy as np
import glob
import os, json
import tqdm 
import argparse
import concurrent.futures

DEFAULT_DATA_PATH = '/work/cadar/Datasets/simulation/test' 
# DEFAULT_DATA_PATH = '/work/cadar/Datasets/simulation/train' 

def load_sample(rgb_path):
    image = cv2.imread(rgb_path)
    segmentation = cv2.imread(rgb_path.replace('rgba', 'segmentation'), cv2.IMREAD_UNCHANGED)

    return {
        'image': image,
        'segmentation': segmentation,
    }

def process_image(image_path):
    mask_path = image_path.replace('rgba', 'bgmask')
    if os.path.exists(mask_path):
        print(f"Mask already exists for {image_path}")
        return

    sample = load_sample(image_path)
    image = sample['image']
    segmentation = sample['segmentation']
    if image is None or segmentation is None:
        print(f"Image or segmentation not found for {image_path}")
        return

    mask = np.zeros(segmentation.shape, dtype=np.uint8)
    mask[segmentation > 1] = 255
    assert (mask>0).sum() == (segmentation > 1).sum(), f"Mask is not correct for {image_path}"
    cv2.imwrite(mask_path, mask)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default=DEFAULT_DATA_PATH, help="Dataset path")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse()
    images = glob.glob(args.data + '/**/rgba*.png', recursive=True)
    print(f"Found {len(images)} images")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm.tqdm(executor.map(process_image, images), total=len(images)))

