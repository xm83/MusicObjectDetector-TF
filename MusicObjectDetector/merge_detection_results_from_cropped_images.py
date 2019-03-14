import argparse
import json
import os
import pandas as pd
from PIL import Image
from PIL.ImageDraw import ImageDraw
from tqdm import tqdm
from joblib import Parallel, delayed

from object_detection.utils.label_map_util import get_label_map_dict


def non_max_suppression(detections_from_previous_crop, all_image_detections):
    keep_items = [True] * all_image_detections.shape[0]
    duplicates = 0
    for index1, row1 in detections_from_previous_crop.iterrows():
        for index2, row2 in all_image_detections.iterrows():
            if row1["class_name"] != row2["class_name"]:
                continue

            intersection_over_union_coefficient = intersection_over_union(
                row1["top"], row1["left"], row1["bottom"], row1["right"],
                row2["top"], row2["left"], row2["bottom"], row2["right"]
            )

            if intersection_over_union_coefficient > 0.5:
                keep_items[index2] = False
                duplicates += 1

    unique_detections = all_image_detections[keep_items]
    # print(f"Removing {duplicates} duplicates.")
    return unique_detections


def intersection_over_union(top1, left1, bottom1, right1, top2, left2, bottom2, right2) -> float:
    intersection_area = intersection(top1, left1, bottom1, right1, top2, left2, bottom2, right2)
    if intersection_area is 0:
        return 0

    area1 = area(top1, left1, bottom1, right1)
    area2 = area(top2, left2, bottom2, right2)
    return intersection_area / (area1 + area2 - intersection_area)


def intersection(top1, left1, bottom1, right1, top2, left2, bottom2, right2):
    x = max(left1, left2)
    y = max(top1, top2)
    w = min(right1, right2) - x
    h = min(bottom1, bottom2) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def area(top, left, bottom, right):
    return (bottom - top) * (right - left)


def merge_results(image_crops):
    all_image_detections = None
    detections_from_previous_crop = pd.DataFrame()
    crops = image_crops["crops"]
    for crop in crops:
        detection_file = os.path.join(detection_results_path, crop["file_name"]).replace(".png", "_detection.csv")
        detections = pd.read_csv(detection_file)
        if detections.size is 0:
            continue
        # Overwrite all image_names
        image_name = os.path.basename(image_crops["image_path"])
        detections["image_name"] = image_name
        # Add offset
        detections["top"] += crop["top_offset"]
        detections["bottom"] += crop["top_offset"]

        unique_detections = non_max_suppression(detections_from_previous_crop, detections)
        if all_image_detections is None:
            all_image_detections = unique_detections
        else:
            all_image_detections = all_image_detections.append(unique_detections)
        detections_from_previous_crop = detections
    all_image_detections.to_csv(os.path.join(merged_results_path, image_name.replace(".png", ".csv")),
                                index=False, float_format="%.2f")
    return all_image_detections

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

classes = {}

def draw_bounding_boxes(image_path, merged_detection_results):
    image = Image.open(image_path)  # type: Image.Image
    image = image.convert("RGB")
    image_draw = ImageDraw(image)
    for i, row in merged_detection_results.iterrows():
        color = STANDARD_COLORS[classes[row["class_name"]] % len(STANDARD_COLORS)]
        image_draw.rectangle([row["left"], row["top"], row["right"], row["bottom"]], outline=color, width=4)
    image.save(image_path.replace(".png", "_detection.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merges the individual object detection results')
    parser.add_argument('--croppings', dest='croppings', type=str, default="data/croppings.json",
                        help='Path to the croppings-file that specifies which image has been cropped and how it has been cropped.')
    parser.add_argument('--detection_results', dest='detection_results_path', type=str, default="detection_output",
                        help='Path to the directory that contains the detection results as individual CSV files')
    parser.add_argument('--class_mapping', dest='class_mapping', type=str, default="mapping_all_classes.txt",
                        help='Path to the class-mapping file that contains the mapping from class-names to unique ids')
    args = parser.parse_args()
    detection_results_path = args.detection_results_path
    merged_results_path = os.path.join(detection_results_path, "merged")
    os.makedirs(merged_results_path, exist_ok=True)

    with open(args.croppings, "r") as file:
        image_crops_list = json.load(file)

    classes = get_label_map_dict(args.class_mapping)

    # Parallel(n_jobs=16)(delayed(merge_results(image_crops)) for image_crops in image_crops_list)
    for image_crops in tqdm(image_crops_list, "Merging detection results"):
        merged_detection_results = merge_results(image_crops)
        draw_bounding_boxes(image_crops["image_path"], merged_detection_results)
