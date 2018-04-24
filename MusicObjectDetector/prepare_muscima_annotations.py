import os
import re
import shutil
from glob import glob
from typing import List

import pandas
from PIL import Image
from lxml import etree
from lxml.etree import Element, SubElement
from muscima.cropobject import CropObject
from omrdatasettools.image_generators.MuscimaPlusPlusImageGenerator import MuscimaPlusPlusImageGenerator
from tqdm import tqdm


def create_annotations_in_pascal_voc_format(annotations_folder: str,
                                            file_name: str,
                                            objects_appearing_in_image: List[CropObject],
                                            image_width: int,
                                            image_height: int,
                                            image_depth: int):
    os.makedirs(annotations_folder, exist_ok=True)

    annotation = Element("annotation")
    folder = SubElement(annotation, "folder")
    folder.text = "muscima_pp_images"
    filename = SubElement(annotation, "filename")
    filename.text = file_name
    source = SubElement(annotation, "source")
    database = SubElement(source, "database")
    database.text = "MUSCIMA++"
    source_annotation = SubElement(source, "annotation")
    source_annotation.text = "MUSCIMA++ (v1.0)"
    image = SubElement(source, "image")
    image.text = "CVC-MUSCIMA"
    size = SubElement(annotation, "size")
    width = SubElement(size, "width")
    width.text = str(image_width)
    height = SubElement(size, "height")
    height.text = str(image_height)
    depth = SubElement(size, "depth")
    depth.text = str(image_depth)

    # Write results to file
    for detected_object in objects_appearing_in_image:
        class_name = detected_object.clsname
        ymin, xmin, ymax, xmax = detected_object.bounding_box

        object = SubElement(annotation, "object")
        name = SubElement(object, "name")
        name.text = class_name
        pose = SubElement(object, "pose")
        pose.text = "Unspecified"
        truncated = SubElement(object, "truncated")
        truncated.text = "0"
        difficult = SubElement(object, "difficult")
        difficult.text = "0"
        bndbox = SubElement(object, "bndbox")
        bb_xmin = SubElement(bndbox, "xmin")
        bb_xmin.text = str(xmin)
        bb_ymin = SubElement(bndbox, "ymin")
        bb_ymin.text = str(ymin)
        bb_xmax = SubElement(bndbox, "xmax")
        bb_xmax.text = str(xmax)
        bb_ymax = SubElement(bndbox, "ymax")
        bb_ymax.text = str(ymax)

    xml_file_path = os.path.join(annotations_folder, os.path.splitext(file_name)[0] + ".xml")
    pretty_xml_string = etree.tostring(annotation, pretty_print=True)

    with open(xml_file_path, "wb") as xml_file:
        xml_file.write(pretty_xml_string)


def prepare_annotations(muscima_image_directory: str,
                        output_path: str,
                        muscima_pp_raw_dataset_directory: str,
                        exported_annotations_file_path: str,
                        annotations_path: str):
    image_paths = glob(muscima_image_directory)
    os.makedirs(output_path, exist_ok=True)

    image_generator = MuscimaPlusPlusImageGenerator()
    raw_data_directory = os.path.join(muscima_pp_raw_dataset_directory, "v1.0", "data", "cropobjects_manual")
    all_xml_files = [y for x in os.walk(raw_data_directory) for y in glob(os.path.join(x[0], '*.xml'))]

    if os.path.exists(exported_annotations_file_path):
        os.remove(exported_annotations_file_path)

    shutil.rmtree(annotations_path, ignore_errors=True)

    for xml_file in tqdm(all_xml_files, desc='Parsing annotation files'):
        crop_objects = image_generator.load_crop_objects_from_xml_file(xml_file)
        doc = crop_objects[0].doc
        result = re.match(r"CVC-MUSCIMA_W-(?P<writer>\d+)_N-(?P<page>\d+)_D-ideal", doc)
        writer = result.group("writer")
        page = result.group("page")

        image_path = None
        for path in image_paths:
            result = re.match(r".*(?P<writer>w-\d+).*(?P<page>p\d+).png", path)
            if ("w-"+writer) == result.group("writer") and ('p' + page.zfill(3)) == result.group("page"):
                image_path = path
                break

        image = Image.open(image_path, "r")  # type: Image.Image
        image_width = image.width
        image_height = image.height
        output_file_path = os.path.join(output_path, "w-{0}_p{1}.jpg".format(writer, page.zfill(3)))
        image.save(output_file_path, "JPEG", quality=95)
        create_annotations_in_pascal_voc_format(annotations_path,
                                                os.path.basename(output_file_path),
                                                crop_objects,
                                                image_width,
                                                image_height,
                                                3)


if __name__ == "__main__":
    dataset_directory = "data"
    muscima_pp_raw_dataset_directory = os.path.join(dataset_directory, "muscima_pp_raw")
    muscima_image_directory = os.path.join(dataset_directory, "cvcmuscima_staff_removal")

    # print("Deleting dataset directory {0}".format(dataset_directory))
    # if os.path.exists(dataset_directory):
    #     shutil.rmtree(dataset_directory, ignore_errors=True)
    #
    # downloader = MuscimaPlusPlusDatasetDownloader(muscima_pp_raw_dataset_directory)
    # downloader.download_and_extract_dataset()
    #
    # downloader = CvcMuscimaDatasetDownloader(muscima_image_directory, CvcMuscimaDataset.StaffRemoval)
    # downloader.download_and_extract_dataset()
    #
    # inverter = ImageInverter()
    # # We would like to work with black-on-white images instead of white-on-black images
    # inverter.invert_images(muscima_image_directory, "*.png")

    prepare_annotations("data/cvcmuscima_staff_removal/*/ideal/*/image/*.png",
                        "data/muscima_pp_images",
                        muscima_pp_raw_dataset_directory,
                        "data/Annotations.csv",
                        "data/Annotations")

    # Create statistics for how many instances of each class exist
    annotations = pandas.read_csv("data/Annotations.csv")
    classes = annotations[['class']].groupby('class').size().reset_index(name='counts')  # type: pandas.DataFrame
    classes.to_csv("data/Class-Statistics.csv", header=True, index=False)
