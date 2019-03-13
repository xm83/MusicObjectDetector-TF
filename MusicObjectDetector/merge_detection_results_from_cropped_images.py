import argparse
import json
import os
import pandas as pd
from tqdm import tqdm


def non_max_suppression(all_image_detections):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merges the individual object detection results')
    parser.add_argument('--croppings', dest='croppings', type=str, default="data/croppings.json",
                        help='Path to the croppings-file that specifies which image has been cropped and how it has been cropped.')
    parser.add_argument('--detection_results', dest='detection_results_path', type=str, default="detection_output",
                        help='Path to the directory that contains the detection results as individual CSV files')
    args = parser.parse_args()
    detection_results_path = args.detection_results_path

    with open(args.croppings, "r") as file:
        image_crops_list = json.load(file)

    for image_crops in tqdm(image_crops_list, "Merging detection results"):
        all_image_detections: pd.DataFrame = None

        original_image_path = image_crops["image_path"]
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
            all_image_detections = pd.concat([all_image_detections, detections])

        non_max_suppression(all_image_detections)
        all_image_detections.to_csv(os.path.join(detection_results_path, image_name.replace(".png", ".csv")),
                                    index=False, float_format="%.2f")
