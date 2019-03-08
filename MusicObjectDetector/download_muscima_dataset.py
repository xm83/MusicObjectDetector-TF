import os
import shutil

from omrdatasettools.converters.ImageColorInverter import ImageColorInverter
from omrdatasettools.downloaders.CvcMuscimaDatasetDownloader import CvcMuscimaDatasetDownloader, CvcMuscimaDataset
from omrdatasettools.downloaders.MuscimaPlusPlusDatasetDownloader import MuscimaPlusPlusDatasetDownloader

if __name__ == "__main__":
    dataset_directory = "data"
    muscima_pp_raw_dataset_directory = os.path.join(dataset_directory, "muscima_pp_raw")
    muscima_image_directory = os.path.join(dataset_directory, "cvcmuscima_staff_removal")

    downloader = MuscimaPlusPlusDatasetDownloader()
    downloader.download_and_extract_dataset(muscima_pp_raw_dataset_directory)

    downloader = CvcMuscimaDatasetDownloader(CvcMuscimaDataset.StaffRemoval)
    downloader.download_and_extract_dataset(muscima_image_directory)

    inverter = ImageColorInverter()
    # We would like to work with black-on-white images instead of white-on-black images
    inverter.invert_images(muscima_image_directory, "*.png")
