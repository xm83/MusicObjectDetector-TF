import os
import shutil

from omrdatasettools.converters.ImageColorInverter import ImageColorInverter
from omrdatasettools.converters.ImageConverter import ImageConverter
from omrdatasettools.downloaders.CvcMuscimaDatasetDownloader import CvcMuscimaDatasetDownloader, CvcMuscimaDataset
from omrdatasettools.downloaders.MuscimaPlusPlusDatasetDownloader import MuscimaPlusPlusDatasetDownloader

if __name__ == "__main__":
    muscima_pp_dataset_directory = os.path.join("data", "muscima_pp")
    muscima_image_directory = os.path.join(muscima_pp_dataset_directory, "v1.0", "data", "images")

    downloader = MuscimaPlusPlusDatasetDownloader()
    downloader.download_and_extract_dataset(muscima_pp_dataset_directory)

    inverter = ImageColorInverter()
    # We would like to work with black-on-white images instead of white-on-black images
    inverter.invert_images(muscima_image_directory, "*.png")
