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


if __name__ == "__main__":
    dataset_directory = "../data"
    muscima_pp_raw_dataset_directory = os.path.join(dataset_directory, "muscima_pp_raw")
    muscima_image_directory = os.path.join(dataset_directory, "cvcmuscima_staff_removal")

    create_statistics_for_full_images(os.path.join(muscima_image_directory, "*/ideal/*/image/*.png"),
                                      muscima_pp_raw_dataset_directory,
                                      "bounding_box_dimensions_absolute.csv",
                                      "bounding_box_dimensions_relative.csv")
