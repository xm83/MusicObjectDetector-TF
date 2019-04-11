import os
from glob import glob

import cv2
import argparse

from PIL import ImageColor
from muscima.io import parse_cropobject_list
from tqdm import tqdm

from inference_over_image import load_category_index

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_bounding_boxes_into_image(image_path: str, ground_truth_annotations_path: str, destination_path: str,
                                   classes_mapping):
    crop_objects = parse_cropobject_list(ground_truth_annotations_path)
    img = cv2.imread(image_path, True)

    for index, crop_object in enumerate(crop_objects):
        # String to float, float to int
        x1 = crop_object.left
        y1 = crop_object.top
        x2 = crop_object.right
        y2 = crop_object.bottom

        color_name = STANDARD_COLORS[classes_mapping[crop_object.clsname] % len(STANDARD_COLORS)]
        color = ImageColor.getrgb(color_name)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        # cv2.putText(img, crop_object.clsname + '/' + str(index + 1), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    cv2.imwrite(destination_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw the bounding boxes from the ground-truth data.')
    parser.add_argument('--image', dest='image', type=str, help='Path to the image (file or directory).',
                        default="data/muscima_pp/v1.0/data/images")
    parser.add_argument('--annotations', dest='annotations', type=str,
                        help='Path to the annotations (file or directory).',
                        default="detection_output/detection_results")
    parser.add_argument('--save_directory', dest='save_directory', type=str,
                        help='Directory, where to save the processed image.',
                        default="detection_output/detection_results")
    parser.add_argument('--label_map', dest='label_map', type=str, default="mapping_all_classes.txt",
                        help='Path to the label map, which is json-file that maps each category name '
                             'to a unique number.')
    args = parser.parse_args()

    category_index = load_category_index(args.label_map, 999999)
    classes_mapping = {v["name"]: v["id"] for k, v in category_index.items()}

    images = []
    if os.path.isfile(args.image):
        images.append(args.image)
    elif os.path.isdir(args.image):
        images.extend(glob(args.image + "/*.png"))

    annotations = []
    if os.path.isfile(args.annotations):
        annotations.append(args.annotations)
    elif os.path.isdir(args.annotations):
        annotations.extend(glob(args.annotations + "/*.xml"))

    output_files = [os.path.join(args.save_directory, os.path.basename(f)) for f in images]
    os.makedirs(args.save_directory, exist_ok=True)

    for image, annotation, output in tqdm(zip(images, annotations, output_files), total=len(output_files),
                                          desc="Drawing annotations"):
        draw_bounding_boxes_into_image(image, annotation, output, classes_mapping)
