# Music Object Detector

This is the repository for the fast and reliable Music Symbol detector with Deep Learning, based on the Tensorflow Object Detection API: 
 
| Original Image | Detection results as training progresses |
|:--------------:|:------------------:|
| ![Original image](MusicObjectDetector/images/crop_undetected.png) | ![Image with detected objects](MusicObjectDetector/images/individualImage1-animation.gif) |
| ![Original image](MusicObjectDetector/images/individualImage3-0.jpg)      | ![Image with detected objects](MusicObjectDetector/images/individualImage3-animated.gif)      |
| ![Original image](MusicObjectDetector/images/individualImage2-0.jpg)      | ![Image with detected objects](MusicObjectDetector/images/individualImage2-animated.gif)      |

The detailed results for various combinations of object-detector, feature-extractor, etc. can be found in [this spreadsheet](https://docs.google.com/spreadsheets/d/174-CnLO-rAoVMst0ngVGHguTlD39ebdxLX9ZLE9Pscw/edit?usp=sharing).


# Preparing the application
This repository contains several scripts that can be used independently of each other. 
Before running them, make sure that you have the necessary requirements installed. 

## Install required libraries

- Python 3.6
- Tensorflow 1.7.0 (or optionally tensorflow-gpu 1.7.0)
- pycocotools (more [infos](https://github.com/matterport/Mask_RCNN/issues/6#issuecomment-341503509))
    - On Linux, run `pip install git+https://github.com/waleedka/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI`
    - On Windows, run `pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI`
- Some libraries, as specified in [requirements.txt](MusicObjectDetector/requirements.txt)

## Prepare this library on Linux

Build the required protobuf files:

```commandline
cd research
protoc object_detection/protos/*.proto --python_out=.
```

## Prepare this library on Windows

Run `DownloadAndBuildProtocolBuffers.ps1` to automate this step or manually build the protobufs by first installing [protocol buffers](https://developers.google.com/protocol-buffers/docs/downloads) and then run:

```commandline
cd research
protoc object_detection/protos/*.proto --python_out=.
```

> Note, that you have to use [version 3.4.0](https://github.com/google/protobuf/releases/download/v3.4.0/) because of a [bug in 3.5.0](https://github.com/google/protobuf/issues/3957)

# Dataset
Run the following scripts to reproduce the dataset locally:

```
# cd into MusicObjectDetector folder
python download_muscima_dataset.py
python prepare_muscima_annotations.py
python DatasetSplitter.py --source_directory=data/muscima_pp_cropped_images_with_stafflines --destination_directory=data/training_validation_test_with_stafflines
python DatasetSplitter.py --source_directory=data/muscima_pp_cropped_images_without_stafflines --destination_directory=data/training_validation_test_without_stafflines
```
  
These scripts will download the datasets automatically, generate cropped images along an Annotation.csv file and split the images into three reproducible parts for training, validation and test. 

Images will be cropped first vertically along the staffs and then horizontally (red boxes) like this (with orange regions overlapping between two regions):
![Cropping of images](MusicObjectDetector/images/w-05_p006_crop_regions.png) 

Now you can create the Tensorflow Records that are required for actually running the training.

```
python create_muscima_tf_record.py --data_dir=data/training_validation_test_with_stafflines --set=training --annotations_dir=Annotations --output_path=data/all_classes_with_staff_lines_writer_independent_split/training.record --label_map_path=mapping_all_classes.txt
python create_muscima_tf_record.py --data_dir=data/training_validation_test_with_stafflines --set=validation --annotations_dir=Annotations --output_path=data/all_classes_with_staff_lines_writer_independent_split/validation.record --label_map_path=mapping_all_classes.txt
python create_muscima_tf_record.py --data_dir=data/training_validation_test_with_stafflines --set=test --annotations_dir=Annotations --output_path=data/all_classes_with_staff_lines_writer_independent_split/test.record --label_map_path=mapping_all_classes.txt

python create_muscima_tf_record.py --data_dir=data/training_validation_test_without_stafflines --set=training --annotations_dir=Annotations --output_path=data/all_classes_without_staff_lines_writer_independent_split/training.record --label_map_path=mapping_all_classes.txt
python create_muscima_tf_record.py --data_dir=data/training_validation_test_without_stafflines --set=validation --annotations_dir=Annotations --output_path=data/all_classes_without_staff_lines_writer_independent_split/validation.record --label_map_path=mapping_all_classes.txt
python create_muscima_tf_record.py --data_dir=data/training_validation_test_without_stafflines --set=test --annotations_dir=Annotations --output_path=data/all_classes_without_staff_lines_writer_independent_split/test.record --label_map_path=mapping_all_classes.txt
```

 If you want to use only a reduced number of classes, you can provide other mappings like `mapping_71_classes.txt`:
 
 ```
 python create_muscima_tf_record.py --data_dir=data/training_validation_test_with_stafflines --set=training --annotations_dir=Annotations --output_path=data/71_classes_with_staff_lines_writer_independent_split/training.record --label_map_path=mapping_71_classes.txt
 python create_muscima_tf_record.py --data_dir=data/training_validation_test_with_stafflines --set=validation --annotations_dir=Annotations --output_path=data/71_classes_with_staff_lines_writer_independent_split/validation.record --label_map_path=mapping_71_classes.txt
 python create_muscima_tf_record.py --data_dir=data/training_validation_test_with_stafflines --set=test --annotations_dir=Annotations --output_path=data/71_classes_with_staff_lines_writer_independent_split/test.record --label_map_path=mapping_71_classes.txt
 
 python create_muscima_tf_record.py --data_dir=data/training_validation_test_without_stafflines --set=training --annotations_dir=Annotations --output_path=data/71_classes_without_staff_lines_writer_independent_split/training.record --label_map_path=mapping_71_classes.txt
 python create_muscima_tf_record.py --data_dir=data/training_validation_test_without_stafflines --set=validation --annotations_dir=Annotations --output_path=data/71_classes_without_staff_lines_writer_independent_split/validation.record --label_map_path=mapping_71_classes.txt
 python create_muscima_tf_record.py --data_dir=data/training_validation_test_without_stafflines --set=test --annotations_dir=Annotations --output_path=data/71_classes_without_staff_lines_writer_independent_split/test.record --label_map_path=mapping_71_classes.txt
 ```
 
## Running the training

### Adding source to Python path
Make sure you have all required folders appended to the [Python path](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#add-libraries-to-pythonpath)

For Linux:
```bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

For Windows (Powershell):
```powershell
$pathToGitRoot = "[GIT_ROOT]"
$pathToSourceRoot = "$($pathToGitRoot)/object_detection"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToSourceRoot);$($pathToGitRoot)/slim"
```

### Adjusting paths
For running the training, you need to change the paths, according to your system

- in the configuration, you want to run, e.g. `configurations/faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_reduced_classes.config`
- if you use them, in the PowerShell scripts in the `training_scripts` folder.

Run the actual training script, by using the pre-defined Powershell scripts in the `training_scripts` folder, or by directly calling

```
# Start the training
python [GIT_ROOT]/research/object_detection/train.py --logtostderr --pipeline_config_path="[GIT_ROOT]/MusicObjectDetector/configurations/[SELECTED_CONFIG].config" --train_dir="[GIT_ROOT]/MusicObjectDetector/data/checkpoints-[SELECTED_CONFIG]-train"

# Start the validation
python [GIT_ROOT]/research/object_detection/eval.py --logtostderr --pipeline_config_path="[GIT_ROOT]/MusicObjectDetector/configurations/[SELECTED_CONFIG].config" --checkpoint_dir="[GIT_ROOT]/MusicObjectDetector/data/checkpoints-[SELECTED_CONFIG]-train" --eval_dir="[GIT_ROOT]/MusicObjectDetector/data/checkpoints-[SELECTED_CONFIG]-validate"
```

A few remarks: The two scripts can and should be run at the same time, to get a live evaluation during the training. The values, may be visualized by calling `tensorboard --logdir=[GIT_ROOT]/MusicObjectDetector/data`.

### Restricting GPU memory usage

Notice that usually Tensorflow allocates the entire memory of your graphics card for the training. In order to run both training and validation at the same time, you might have to restrict Tensorflow from doing so, by opening `train.py` and `eval.py` and uncomment the respective (prepared) lines in the main function. E.g.:

```
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
```

### Training with pre-trained weights

It is recommended that you use pre-trained weights for known networks to speed up training and improve overall results. To do so, head over to the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), download and unzip the respective trained model, e.g. `faster_rcnn_inception_resnet_v2_atrous_coco` for reproducing the best results, we obtained. The path to the unzipped files, must be specified inside of the configuration in the `train_config`-section, e.g.

```
train-config: {
  fine_tune_checkpoint: "C:/Users/Alex/Repositories/MusicObjectDetector-TF/MusicObjectDetector/data/faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08/model.ckpt"
  from_detection_checkpoint: true
}
```

> Note that inside that folder, there is no actual file, called `model.ckpt`, but multiple files called `model.ckpt.[something]`.

# Dimension clustering

For optimizing the performance of the detector, we adopted the dimensions clustering algorithm, proposed in the [YOLO 9000 paper](https://arxiv.org/abs/1612.08242).
While preparing the dataset, the `muscima_image_cutter.py` script created a file called `Annotations.csv` and a folder called `Annotations`. 
Both will contain the same annotations, but in different formats. While the csv-file contains all annotations in a plain list, the Annotations
folder contains one xml-file per image, complying with the format used for the [Pascal VOC project](http://host.robots.ox.ac.uk/pascal/VOC/).

To perform dimension clustering on the cropped images, run the following scripts:
```
python generate_muscima_statistics.py
python muscima_dimension_clustering.py
```
The first script will load all annotations and create four csv-files containing the dimensions for each annotation 
from all images, including their relative sizes, compared to the entire image.
The second script loads those statistics and performs dimension clustering, use a k-means algorithm on the relative 
dimensions of annotations.   

# License

Published under MIT License,

Copyright (c) 2018 [Alexander Pacha](http://alexanderpacha.com), [TU Wien](https://www.ims.tuwien.ac.at/people/alexander-pacha) and Kwon-Young Choi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
