import os
import re
from glob import glob
from typing import Tuple, List

import numpy
import pandas
from PIL import Image
from muscima.cropobject import CropObject
from omrdatasettools.image_generators.MuscimaPlusPlusImageGenerator import MuscimaPlusPlusImageGenerator
from tqdm import tqdm


def create_statistics_for_full_images(glob_pattern_for_retrieving_muscima_images: str,
                                      muscima_pp_raw_data_directory: str,
                                      exported_absolute_dimensions_file_path: str,
                                      exported_relative_dimensions_file_path: str):
    image_paths = glob(glob_pattern_for_retrieving_muscima_images)

    if os.path.exists(exported_absolute_dimensions_file_path):
        os.remove(exported_absolute_dimensions_file_path)

    if os.path.exists(exported_relative_dimensions_file_path):
        os.remove(exported_relative_dimensions_file_path)

    image_generator = MuscimaPlusPlusImageGenerator()
    all_xml_files = image_generator.get_all_xml_file_paths(muscima_pp_raw_data_directory)

    absolute_dimensions = []
    relative_dimensions = []
    for xml_file in tqdm(all_xml_files, desc='Parsing annotation files'):
        crop_objects = image_generator.load_crop_objects_from_xml_file(xml_file)

        doc = crop_objects[0].doc
        result = re.match(r"CVC-MUSCIMA_W-(?P<writer>\d+)_N-(?P<page>\d+)_D-ideal", doc)
        writer = result.group("writer")
        page = "0" + result.group("page")

        for image_path in image_paths:
            image_path_result = re.match(r".*w-(?P<writer>\d+).*p(?P<page>\d+).png", image_path)
            image_path_writer = image_path_result.group("writer")
            image_path_page = image_path_result.group("page")
            if image_path_writer == writer and image_path_page == page:
                path = image_path
                break
        
        image = Image.open(path, "r")  # type: Image.Image
        image_width = image.width
        image_height = image.height

        for crop_object in crop_objects:
            class_name = crop_object.clsname
            top, left, bottom, right = crop_object.bounding_box
            width = right - left
            height = bottom - top
            x_center = width / 2.0 + left
            y_center = height / 2.0 + top

            absolute_dimensions.append([class_name, left, right, top, bottom, x_center, y_center, width, height])
            relative_dimensions.append([class_name, left / image_width, right / image_width,
                                        top / image_height, bottom / image_height,
                                        x_center / image_width, y_center / image_height,
                                        width / image_width, height / image_height])

    absolute_statistics = pandas.DataFrame(absolute_dimensions, columns=["class","xmin","xmax","ymin","ymax","x_c","y_c","width","height"])
    absolute_statistics.to_csv(exported_absolute_dimensions_file_path, float_format="%.5f", index=False)
    relative_statistics = pandas.DataFrame(relative_dimensions, columns=["class","xmin","xmax","ymin","ymax","x_c","y_c","width","height"])
    relative_statistics.to_csv(exported_relative_dimensions_file_path, float_format="%.5f", index=False)


def create_statistics_for_cropped_images(annotations_csv,
                                         path_to_cropped_images,
                                         exported_absolute_dimensions_file_path,
                                         exported_relative_dimensions_file_path):
    annotations = pandas.read_csv(annotations_csv)
    cropped_image_dimensions = []
    for filename in tqdm(os.listdir(path_to_cropped_images)):
        image = Image.open(os.path.join(path_to_cropped_images, filename))  # type: Image.Image
        cropped_image_dimensions.append((filename, image.width, image.height))

    width = (annotations['right'] - annotations['left']).rename('width')
    height = (annotations['bottom'] - annotations['top']).rename('height')
    x_center = (annotations['left'] + width / 2).rename('x_center')
    y_center = (annotations['top'] + height / 2).rename('y_center')

    absolute_annotations = pandas.concat(
        [annotations[['filename', 'class', 'left', 'right', 'top', 'bottom']], x_center, y_center, width, height],
        axis=1)  # type: pandas.DataFrame
    absolute_annotations.to_csv(exported_absolute_dimensions_file_path, float_format="%.5f", index=False)

    # Compute relative annotations
    image_dimensions = pandas.DataFrame(cropped_image_dimensions, columns=['filename', 'image_width', 'image_height'])
    joined = annotations.merge(image_dimensions, on='filename')
    left = (joined['left'] / joined['image_width']).rename('left')
    right = (joined['right'] / joined['image_width']).rename('right')
    top = (joined['top'] / joined['image_height']).rename('top')
    bottom = (joined['bottom'] / joined['image_height']).rename('bottom')
    x_center = (x_center / joined['image_width']).rename('x_center')
    y_center = (y_center / joined['image_height']).rename('y_center')
    width = (width / joined['image_width']).rename('width')
    height = (height / joined['image_height']).rename('height')

    relative_annotations = pandas.concat(
        [joined[['filename', 'class']], left, right, top, bottom, x_center, y_center, width, height],
        axis=1)  # type: pandas.DataFrame
    relative_annotations.to_csv(exported_relative_dimensions_file_path, float_format="%.5f", index=False)


if __name__ == "__main__":
    dataset_directory = "data"
    muscima_pp_raw_dataset_directory = os.path.join(dataset_directory, "muscima_pp_raw")
    muscima_image_directory = os.path.join(dataset_directory, "cvcmuscima_staff_removal")

    create_statistics_for_full_images("data/cvcmuscima_staff_removal/*/ideal/*/image/*.png", "data/muscima_pp_raw",
                                      "data/bounding_box_dimensions_absolute.csv",
                                      "data/bounding_box_dimensions_relative.csv")

    create_statistics_for_cropped_images(annotations_csv="data/Annotations.csv",
                                         path_to_cropped_images="data/muscima_pp_cropped_images_with_stafflines",
                                         exported_absolute_dimensions_file_path="data/bounding_box_dimensions_cropped_images_absolute.csv",
                                         exported_relative_dimensions_file_path="data/bounding_box_dimensions_cropped_images_relative.csv")
