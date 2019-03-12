import os
import re
import shutil
from glob import glob
from typing import Tuple, List

import pandas
from PIL import Image
from muscima.cropobject import CropObject
from omrdatasettools.image_generators.MuscimaPlusPlusImageGenerator import MuscimaPlusPlusImageGenerator
from tqdm import tqdm

from muscima_annotation_generator import create_annotations_in_pascal_voc_format


def cut_images(muscima_image_directory: str, output_path: str, muscima_pp_raw_dataset_directory: str,
               exported_annotations_file_path: str, annotations_path: str):
    image_paths = glob(muscima_image_directory)
    os.makedirs(output_path, exist_ok=True)

    image_generator = MuscimaPlusPlusImageGenerator()
    raw_data_directory = os.path.join(muscima_pp_raw_dataset_directory, "v1.0", "data", "cropobjects_withstaff")
    all_xml_files = [y for x in os.walk(raw_data_directory) for y in glob(os.path.join(x[0], '*.xml'))]

    if os.path.exists(exported_annotations_file_path):
        os.remove(exported_annotations_file_path)

    shutil.rmtree(annotations_path, ignore_errors=True)

    crop_object_annotations: List[Tuple[str, str, List[CropObject]]] = []

    for xml_file in tqdm(all_xml_files, desc='Parsing annotation files'):
        crop_objects = image_generator.load_crop_objects_from_xml_file(xml_file)
        doc = crop_objects[0].doc
        result = re.match(r"CVC-MUSCIMA_W-(?P<writer>\d+)_N-(?P<page>\d+)_D-ideal", doc)
        writer = result.group("writer")
        page = result.group("page")
        crop_object_annotations.append(('w-' + writer, 'p' + page.zfill(3), crop_objects))

    crop_annotations = []
    for image_path in tqdm(image_paths, desc="Cutting images"):
        result = re.match(r".*(?P<writer>w-\d+).*(?P<page>p\d+).png", image_path)
        writer = result.group("writer")
        page = result.group("page")
        image = Image.open(image_path, "r")  # type: Image.Image
        image_width = image.width
        image_height = image.height
        objects_appearing_in_image: List[CropObject] = None
        for crop_object_annotation in crop_object_annotations:
            if writer == crop_object_annotation[0] and page == crop_object_annotation[1]:
                objects_appearing_in_image = crop_object_annotation[2]
                break

        if objects_appearing_in_image is None:
            # Image has annotated staff-lines, but does not have corresponding crop-object annotations, so skip it
            continue

        staff_objects = [x for x in objects_appearing_in_image if x.clsname == "staff"]
        max_offset_before_first_and_after_last_staff = 120

        if staff_objects is None:
            # Image has no staff lines -> Report error
            print("Error: Image {0} has no annotated staff lines".format(image_path))
            continue

        next_y_top = max(0, staff_objects[0].top - max_offset_before_first_and_after_last_staff)
        last_bottom = min(staff_objects[len(staff_objects) - 1].bottom + max_offset_before_first_and_after_last_staff,
                          image_height)

        output_image_counter = 1
        for staff_index in range(len(staff_objects)):
            staff = staff_objects[staff_index]
            if staff_index < len(staff_objects) - 1:
                y_bottom = staff_objects[staff_index + 1].top
            else:
                y_bottom = last_bottom
            y_top = next_y_top
            next_y_top = staff.bottom

            image_crop_bounding_box_left_top_bottom_right = (0, y_top, image_width, y_bottom)
            image_crop_bounding_box_top_left_bottom_right = (y_top, 0, y_bottom, image_width)

            file_name = "{0}_{1}_{2}.jpg".format(writer, page, output_image_counter)
            output_image_counter += 1

            objects_appearing_in_cropped_image = \
                compute_objects_appearing_in_cropped_image(file_name,
                                                           image_crop_bounding_box_top_left_bottom_right,
                                                           objects_appearing_in_image)

            cropped_image = image.crop(image_crop_bounding_box_left_top_bottom_right).convert('RGB')

            for object_appearing_in_cropped_image in objects_appearing_in_cropped_image:
                file_name = object_appearing_in_cropped_image[0]
                class_name = object_appearing_in_cropped_image[1]
                translated_bounding_box = object_appearing_in_cropped_image[2]
                trans_top, trans_left, trans_bottom, trans_right = translated_bounding_box
                crop_annotations.append([file_name, trans_left, trans_top, trans_right, trans_bottom, class_name])

            create_annotations_in_pascal_voc_format(annotations_path, file_name, objects_appearing_in_cropped_image,
                                                    cropped_image.width, cropped_image.height, 3)

            output_file = os.path.join(output_path, file_name)
            cropped_image.save(output_file, "png")

    annotation_data = pandas.DataFrame(crop_annotations,
                                       columns=['filename', 'left', 'top', 'right', 'bottom', 'class'])
    annotation_data.to_csv(exported_annotations_file_path, index=False)


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def area(a):
    top, left, bottom, right = a
    return (bottom - top) * (right - left)


def compute_objects_appearing_in_cropped_image(file_name: str,
                                               image_crop_bounding_box_top_left_bottom_right: Tuple[int, int, int, int],
                                               all_music_objects_appearing_in_image: List[CropObject]) \
        -> List[Tuple[str, str, Tuple[int, int, int, int]]]:
    x_translation_for_cropped_image = image_crop_bounding_box_top_left_bottom_right[1]
    y_translation_for_cropped_image = image_crop_bounding_box_top_left_bottom_right[0]

    objects_appearing_in_cropped_image: List[Tuple[str, str, Tuple[int, int, int, int]]] = []
    for music_object in all_music_objects_appearing_in_image:
        if music_object.clsname in ["staff", "staff_line", "staff_space"]:
            continue

        intersection_over_area = intersection(image_crop_bounding_box_top_left_bottom_right,
                                              music_object.bounding_box) / area(
            music_object.bounding_box)
        if intersection_over_area > 0.8:
            top, left, bottom, right = music_object.bounding_box
            img_top, img_left, img_bottom, img_right = image_crop_bounding_box_top_left_bottom_right
            img_width = img_right - img_left - 1
            img_height = img_bottom - img_top - 1
            translated_bounding_box = (
                max(0, top - y_translation_for_cropped_image),
                max(0, left - x_translation_for_cropped_image),
                min(img_height, bottom - y_translation_for_cropped_image),
                min(img_width, right - x_translation_for_cropped_image))
            objects_appearing_in_cropped_image.append((file_name, music_object.clsname, translated_bounding_box))

    return objects_appearing_in_cropped_image


if __name__ == "__main__":
    dataset_directory = "data"
    muscima_pp_raw_dataset_directory = os.path.join(dataset_directory, "muscima_pp_raw")
    muscima_image_directory = os.path.join(dataset_directory, "cvcmuscima_staff_removal")

    cut_images("data/cvcmuscima_staff_removal/*/ideal/*/image/*.png",
               "data/muscima_pp_cropped_images_with_stafflines",
               muscima_pp_raw_dataset_directory,
               "data/Annotations.csv",
               "data/Annotations")

    # Create statistics for how many instances of each class exist
    annotations = pandas.read_csv("data/Annotations.csv")
    classes = annotations[['class']].groupby('class').size().reset_index(name='counts')  # type: pandas.DataFrame
    classes.to_csv("data/Class-Statistics.csv", header=True, index=False)
