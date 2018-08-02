import os
import re
from glob import glob

from PIL import Image
from tqdm import tqdm
import pandas as pd


def create_statistics_for_full_images(dataset_directory:str,
                                      annotations_csv: str,
                                      exported_absolute_dimensions_file_path: str,
                                      exported_relative_dimensions_file_path: str):

    if os.path.exists(exported_absolute_dimensions_file_path):
        os.remove(exported_absolute_dimensions_file_path)

    if os.path.exists(exported_relative_dimensions_file_path):
        os.remove(exported_relative_dimensions_file_path)

    absolute_dimensions = []
    relative_dimensions = []

    annotations = pd.read_csv(annotations_csv)
    image_dimensions_dictionary = {}

    for index, row in tqdm(annotations.iterrows(), desc='Converting annotations', total=annotations.shape[0]):
        image_path = os.path.join(dataset_directory, row['path_to_image'])
        if image_path not in image_dimensions_dictionary:
            image = Image.open(image_path, "r")  # type: Image.Image
            image_width = image.width
            image_height = image.height
            image_dimensions_dictionary[image_path] = (image_width, image_height)
        else:
            image_width, image_height = image_dimensions_dictionary[image_path]

        class_name = row['class_name']
        top, left, bottom, right = row['top'], row['left'], row['bottom'], row['right']
        width = right - left
        height = bottom - top
        x_center = width / 2.0 + left
        y_center = height / 2.0 + top

        absolute_dimensions.append([class_name, left, right, top, bottom, x_center, y_center, width, height])
        relative_dimensions.append(
            [class_name, left / image_width, right / image_width, top / image_height, bottom / image_height,
             x_center / image_width, y_center / image_height,
             width / image_width, height / image_height])

    absolute_statistics = pd.DataFrame(absolute_dimensions,
                                       columns=["class", "xmin", "xmax", "ymin", "ymax", "x_c", "y_c", "width",
                                                "height"])
    absolute_statistics.to_csv(exported_absolute_dimensions_file_path, float_format="%.5f", index=False)
    relative_statistics = pd.DataFrame(relative_dimensions,
                                       columns=["class", "xmin", "xmax", "ymin", "ymax", "x_c", "y_c", "width",
                                                "height"])
    relative_statistics.to_csv(exported_relative_dimensions_file_path, float_format="%.5f", index=False)


if __name__ == "__main__":

    path_to_normalized_datasets = "../data/normalized/"
    datasets = ['deepscores', 'mensural', 'muscima']

    for dataset in datasets:
        dataset_directory = path_to_normalized_datasets + dataset

        annotations_csv = os.path.join(dataset_directory, "annotations.csv")

        create_statistics_for_full_images(dataset_directory,
                                          annotations_csv,
                                          dataset + "_bounding_box_dimensions_absolute.csv",
                                          dataset + "_bounding_box_dimensions_relative.csv")
