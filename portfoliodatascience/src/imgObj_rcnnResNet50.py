'''
Applying ResNet50 (public convolutional neuronal network) to label image features.
'''
import argparse
import sys
import numpy as np

from keras.applications.resnet50 import ResNet50, decode_predictions
from keras.preprocessing.image import load_img, img_to_array

SLICE_SIZE = 224
DEFAULT_THRESHOLD = 0.4

def get_slices(img_filename):
    """loads an image and cuts it into slices of 224 x224 pixels"""
    img = img_to_array(load_img(img_filename))
    
    slice_x = img.shape[0] // SLICE_SIZE
    slice_y = img.shape[1] // SLICE_SIZE
        
    slices = []
    for i in range(slice_x):
        for j in range(slice_y):
            section = img[SLICE_SIZE * i:SLICE_SIZE * (i + 1), SLICE_SIZE * j:SLICE_SIZE * (j + 1)]
            slices.append(section)

    print("generated", len(slices), "slices")
    return np.array(slices)

def print_predictions(filename, preds, threshold=DEFAULT_THRESHOLD):
    """prints tabular output: filename, label, prob"""
    y_pred = decode_predictions(preds, top=3)
    for lev1 in y_pred:
        for _, label, prob in lev1:
            if prob > threshold:
                print(f"{filename}, {label},{prob}")
    
def label_images(all_filenames, threshold):
    """Slice an image and predict classes for each"""
    model = ResNet50()
    for filename in all_filenames:
        slices = get_slices(filename)
        if slices.shape != (0,):
            preds = model.predict(slices)
            print_predictions(filename, preds, threshold)
        else:
            print(f"{filename} is too small (Min. Requirement: {SLICE_SIZE}x{SLICE_SIZE} pixels) ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image labeling pipeline.')

    parser.add_argument('-t', '--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='softmax probability threshold')

    parser.add_argument('-o', '--outfile', nargs='?', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help="output file")

    parser.add_argument('image', type=str, nargs='+',
                        help='image filename(s)')

    args = parser.parse_args()
    label_images(args.image, args.threshold)
