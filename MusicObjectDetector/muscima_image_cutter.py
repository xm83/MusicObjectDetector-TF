import os
import re
import shutil
from glob import glob
from typing import Tuple, List

from PIL import Image, ImageDraw
from muscima.cropobject import CropObject
from omrdatasettools.converters.ImageInverter import ImageInverter
from omrdatasettools.downloaders.CvcMuscimaDatasetDownloader import CvcMuscimaDatasetDownloader, CvcMuscimaDataset
from omrdatasettools.downloaders.MuscimaPlusPlusDatasetDownloader import MuscimaPlusPlusDatasetDownloader
from omrdatasettools.image_generators.MuscimaPlusPlusImageGenerator import MuscimaPlusPlusImageGenerator
from tqdm import tqdm
import pandas

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
            previous_width = 0
            overlap = 100
            for crop_width in range(500, 3501, 500):

                if crop_width > image_width:
                    crop_width = image_width
                image_crop_bounding_box = (previous_width, y_top, crop_width, y_bottom)
                image_crop_bounding_box_top_left_bottom_right = (y_top, previous_width, y_bottom, crop_width)
                previous_width = crop_width - overlap

                file_name = "{0}_{1}_{2}.jpg".format(writer, page, output_image_counter)
                output_image_counter += 1

                objects_appearing_in_cropped_image = \
                    compute_objects_appearing_in_cropped_image(file_name,
                                                               image_crop_bounding_box_top_left_bottom_right,
                                                               objects_appearing_in_image)

                cropped_image = image.crop(image_crop_bounding_box).convert('RGB')

                for object_appearing_in_cropped_image in objects_appearing_in_cropped_image:
                    file_name = object_appearing_in_cropped_image[0]
                    class_name = object_appearing_in_cropped_image[1]
                    translated_bounding_box = object_appearing_in_cropped_image[2]
                    trans_top, trans_left, trans_bottom, trans_right = translated_bounding_box
                    crop_annotations.append([file_name, trans_left, trans_top, trans_right, trans_bottom, class_name])

                create_annotations_in_pascal_voc_format(annotations_path, file_name, objects_appearing_in_cropped_image,
                                                        cropped_image.width, cropped_image.height, 3)

                # draw_bounding_boxes(cropped_image, objects_appearing_in_cropped_image)
                output_file = os.path.join(output_path, file_name)
                cropped_image.save(output_file, "JPEG", quality=95)

    annotation_data = pandas.DataFrame(crop_annotations, columns=['filename', 'left', 'top', 'right', 'bottom', 'class'])
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
        # crop_fully_contains_bounding_box = bounding_box_in(image_crop_bounding_box, music_object.bounding_box)
        # if crop_fully_contains_bounding_box:
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


def bounding_box_in(image_crop_bounding_box: Tuple[int, int, int, int],
                    crop_object_bounding_box: Tuple[int, int, int, int]) -> bool:
    image_left, image_top, image_right, image_bottom = image_crop_bounding_box
    object_top, object_left, object_bottom, object_right = crop_object_bounding_box
    if object_left >= image_left and object_right <= image_right \
            and object_top >= image_top and object_bottom <= image_bottom:
        return True
    return False


def draw_bounding_boxes(cropped_image: Image,
                        objects_appearing_in_cropped_image: List[Tuple[str, Tuple[int, int, int, int]]]):
    draw = ImageDraw.Draw(cropped_image)
    red = (255, 0, 0)
    for object_in_image in objects_appearing_in_cropped_image:
        top, left, bottom, right = object_in_image[1]
        draw.rectangle((left, top, right, bottom), fill=None, outline=red)


def delete_unused_images(muscima_image_directory: str):
    """ We only need the images of the ideal scores, so we can delete all other images from the dataset
        that are not inside the ideal/w-xx/image/ directory
    """
    all_image_paths = [y for x in os.walk(muscima_image_directory) for y in glob(os.path.join(x[0], '*.png'))]

    for image_path in tqdm(all_image_paths, desc="Deleting unused images"):
        if not ('ideal' in image_path and ('image' in image_path or 'symbol' in image_path)):
            os.remove(image_path)


if __name__ == "__main__":
    dataset_directory = "data"
    muscima_pp_raw_dataset_directory = os.path.join(dataset_directory, "muscima_pp_raw")
    muscima_image_directory = os.path.join(dataset_directory, "cvcmuscima_staff_removal")

    print("Deleting dataset directory {0}".format(dataset_directory))
    if os.path.exists(dataset_directory):
        shutil.rmtree(dataset_directory, ignore_errors=True)

    downloader = MuscimaPlusPlusDatasetDownloader(muscima_pp_raw_dataset_directory)
    downloader.download_and_extract_dataset()

    downloader = CvcMuscimaDatasetDownloader(muscima_image_directory, CvcMuscimaDataset.StaffRemoval)
    downloader.download_and_extract_dataset()

    delete_unused_images(muscima_image_directory)

    inverter = ImageInverter()
    # We would like to work with black-on-white images instead of white-on-black images
    inverter.invert_images(muscima_image_directory, "*.png")

    cut_images("data/cvcmuscima_staff_removal/*/ideal/*/image/*.png",
               "data/muscima_pp_cropped_images_with_stafflines",
               muscima_pp_raw_dataset_directory,
               "data/Annotations.csv",
               "data/Annotations")
    cut_images("data/cvcmuscima_staff_removal/*/ideal/*/symbol/*.png",
               "data/muscima_pp_cropped_images_without_stafflines",
               muscima_pp_raw_dataset_directory,
               "data/Annotations.csv",
               "data/Annotations")

    # Create statistics for how many instances of each class exist
    annotations = pandas.read_csv("data/Annotations.csv")
    classes = annotations[['class']].groupby('class').size().reset_index(name='counts') #type: pandas.DataFrame
    classes.to_csv("data/Class-Statistics.csv", header=True, index=False)