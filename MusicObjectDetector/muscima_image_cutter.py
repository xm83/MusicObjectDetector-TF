import glob
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

from muscima_annotation_generator import create_annotations_in_plain_format, create_annotations_in_pascal_voc_format


def cut_images(muscima_image_directory: str, staff_vertical_positions_file: str, output_path: str,
               muscima_pp_raw_data_directory: str, exported_annotations_file_path: str, annotations_path: str):
    image_paths = [y for x in os.walk(muscima_image_directory) for y in glob(os.path.join(x[0], '*.png'))]
    os.makedirs(output_path, exist_ok=True)

    image_generator = MuscimaPlusPlusImageGenerator()
    all_xml_files = image_generator.get_all_xml_file_paths(muscima_pp_raw_data_directory)

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

    with open(staff_vertical_positions_file) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    annotations = [x.strip().split(';') for x in content]

    images_to_cut: List[Tuple[str, str, str, str]] = []

    for image_path in image_paths:
        result = re.match(r".*(?P<writer>w-\d+).*(?P<page>p\d+).png", image_path)
        writer = result.group("writer")
        page = result.group("page")
        coordinates = None
        for annotation in annotations:
            if annotation[0] == writer and annotation[1] == page:
                coordinates = annotation[2]
                break

        if coordinates is not None:
            images_to_cut.append((image_path, writer, page, coordinates))

    for image_to_cut in tqdm(images_to_cut, desc="Cutting images"):
        path, writer, page, coordinates = image_to_cut
        staff_line_pairs = coordinates.split(',')
        image = Image.open(path, "r")  # type: Image.Image
        width = image.width
        objects_appearing_in_image: List[CropObject] = None
        for crop_object_annotation in crop_object_annotations:
            if writer == crop_object_annotation[0] and page == crop_object_annotation[1]:
                objects_appearing_in_image = crop_object_annotation[2]
                break

        if objects_appearing_in_image is None:
            # Image has annotated staff-lines, but does not have corresponding crop-object annotations, so skip it
            continue

        i = 1
        for pair in staff_line_pairs:
            y_top, y_bottom = map(int, pair.split(':'))
            previous_width = 0
            overlap = 100
            for crop_width in range(500, 3501, 500):

                if crop_width > width:
                    crop_width = width
                image_crop_bounding_box = (previous_width, y_top, crop_width, y_bottom)
                image_crop_bounding_box_top_left_bottom_right = (y_top, previous_width, y_bottom, crop_width)
                previous_width = crop_width - overlap

                file_name = "{0}_{1}_{2}.jpg".format(writer, page, i)
                i += 1

                objects_appearing_in_cropped_image = \
                    compute_objects_appearing_in_cropped_image(file_name,
                                                               image_crop_bounding_box_top_left_bottom_right,
                                                               objects_appearing_in_image)

                cropped_image = image.crop(image_crop_bounding_box).convert('RGB')

                create_annotations_in_plain_format(exported_annotations_file_path, objects_appearing_in_cropped_image)
                create_annotations_in_pascal_voc_format(annotations_path, file_name, objects_appearing_in_cropped_image,
                                                        cropped_image.width, cropped_image.height, 3)

                # draw_bounding_boxes(cropped_image, objects_appearing_in_cropped_image)
                output_file = os.path.join(output_path, file_name)
                cropped_image.save(output_file, "JPEG", quality=95)


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
            translated_bounding_box = (
                top - y_translation_for_cropped_image, left - x_translation_for_cropped_image,
                bottom - y_translation_for_cropped_image, right - x_translation_for_cropped_image)
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
        if not ('ideal' in image_path and 'image' in image_path):
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
    #
    shutil.copy("Staff-Vertical-Positions.txt", dataset_directory)

    cut_images("data/cvcmuscima_staff_removal", "data/Staff-Vertical-Positions.txt",
               "data/muscima_pp_cropped_images", "data/muscima_pp_raw", "data/Annotations.txt", "data/Annotations")
