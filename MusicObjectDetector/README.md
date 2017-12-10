# Music Object Detector

This is the repository for the fast and reliable Music Symbol detector with Deep Learning, based on the Tensorflow Object Detection API: 
 
![Original image](images/crop_undetected.png) 

![Image with detected objects](images/crop_partially_detected.png)

The detailed results for various combinations of object-detector, feature-extractor, etc. can be found in [this spreadsheet](https://docs.google.com/spreadsheets/d/174-CnLO-rAoVMst0ngVGHguTlD39ebdxLX9ZLE9Pscw/edit?usp=sharing).


# Running the application
This repository contains several scripts that can be used independently of each other. 
Before running them, make sure that you have the necessary requirements installed. 

## Requirements

- Python 3.6
- Tensorflow 1.4.0 (or optionally tensorflow-gpu 1.4.0)

For installing Tensorflow and Keras we recommend using [Anaconda](https://www.continuum.io/downloads) or 
[Miniconda](https://conda.io/miniconda.html) as Python distribution (we did so for preparing Travis-CI and it worked).

To accelerate training even further, you can make use of your GPU, by installing tensorflow-gpu instead of tensorflow
via pip (note that you can only have one of them) and the required Nvidia drivers. For Windows, we recommend the
[excellent tutorial by Phil Ferriere](https://github.com/philferriere/dlwin). For Linux, we recommend using the
 official tutorials by [Tensorflow](https://www.tensorflow.org/install/) and [Keras](https://keras.io/#installation).

### Linux

See https://github.com/tensorflow/models for information 

Build the required libraries:

```commandline
cd research
protoc object_detection/protos/*.proto --python_out=.
cd slim
python setup.py install
cd ..
python setup.py install
```

### Windows
First, make sure you have [protocol buffers](https://developers.google.com/protocol-buffers/docs/downloads) installed, by heading over to [the download page](https://github.com/google/protobuf/releases/tag/v2.6.0) and download the version 2.6.0. Extract and copy the protoc.exe to a place, where you can run it from later on.  

```commandline
cd research
protoc object_detection/protos/*.proto --python_out=.
```
if protoc does not understand the *-operator, build the files individually:
```commandline
protoc object_detection\protos\anchor_generator.proto               --python_out=.
protoc object_detection\protos\argmax_matcher.proto                 --python_out=.
protoc object_detection\protos\bipartite_matcher.proto              --python_out=.
protoc object_detection\protos\box_coder.proto                      --python_out=.
protoc object_detection\protos\box_predictor.proto                  --python_out=.
protoc object_detection\protos\eval.proto                           --python_out=.
protoc object_detection\protos\faster_rcnn.proto                    --python_out=.
protoc object_detection\protos\faster_rcnn_box_coder.proto          --python_out=.
protoc object_detection\protos\grid_anchor_generator.proto          --python_out=.
protoc object_detection\protos\hyperparams.proto                    --python_out=.
protoc object_detection\protos\image_resizer.proto                  --python_out=.
protoc object_detection\protos\input_reader.proto                   --python_out=.
protoc object_detection\protos\keypoint_box_coder.proto             --python_out=.
protoc object_detection\protos\losses.proto                         --python_out=.
protoc object_detection\protos\matcher.proto                        --python_out=.
protoc object_detection\protos\mean_stddev_box_coder.proto          --python_out=.
protoc object_detection\protos\model.proto                          --python_out=.
protoc object_detection\protos\optimizer.proto                      --python_out=.
protoc object_detection\protos\pipeline.proto                       --python_out=.
protoc object_detection\protos\post_processing.proto                --python_out=.
protoc object_detection\protos\preprocessor.proto                   --python_out=.
protoc object_detection\protos\region_similarity_calculator.proto   --python_out=.
protoc object_detection\protos\square_box_coder.proto               --python_out=.
protoc object_detection\protos\ssd.proto                            --python_out=.
protoc object_detection\protos\ssd_anchor_generator.proto           --python_out=.
protoc object_detection\protos\string_int_label_map.proto           --python_out=.
protoc object_detection\protos\train.proto                          --python_out=.
```

Install the python packages
```commandline
cd slim
python setup.py install
cd ..
python setup.py install
```

> If you get the exception `error: could not create 'build': Cannot create a file when that file already exists` here, delete the `BUILD` file inside first

Now add the [source to the python path](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#add-libraries-to-pythonpath) or just copy the `object_detection` folder and the `slim` folder into your `[Anaconda3]/Lib/site-packages` directory. 


# Dataset
If you are just interested in the dataset, the split and the annotations used in this project, you can run the following scripts to reproduce the dataset locally:

```
# cd into MusicObjectDetector folder
python generate_mapping.py data/muscima_pp_raw/v0.9.1/data/cropobjects mapping_reduced_class2.txt
python muscima_image_cutter.py
python DatasetSplitter.py --source_directory=data/muscima_pp_cropped_images_with_stafflines --destination_directory=data/training_validation_test_with_stafflines
python DatasetSplitter.py --source_directory=data/muscima_pp_cropped_images_without_stafflines --destination_directory=data/training_validation_test_without_stafflines

(cd ../research; python ../MusicObjectDetector/create_muscima_tf_record.py --data_dir=../MusicObjectDetector/data/training_validation_test_with_stafflines --set=training --annotations_dir=Annotations --output_path=../MusicObjectDetector/data/training_validation_test_with_stafflines/training.record --label_map_path=../MusicObjectDetector/mapping_reduced_class2.txt)
(cd ../research; python ../MusicObjectDetector/create_muscima_tf_record.py --data_dir=../MusicObjectDetector/data/training_validation_test_with_stafflines --set=validation --annotations_dir=Annotations --output_path=../MusicObjectDetector/data/training_validation_test_with_stafflines/validation.record --label_map_path=../MusicObjectDetector/mapping_reduced_class2.txt)
(cd ../research; python ../MusicObjectDetector/create_muscima_tf_record.py --data_dir=../MusicObjectDetector/data/training_validation_test_with_stafflines --set=test --annotations_dir=Annotations --output_path=../MusicObjectDetector/data/training_validation_test_with_stafflines/test.record --label_map_path=../MusicObjectDetector/mapping_reduced_class2.txt)

(cd ../research; python ../MusicObjectDetector/create_muscima_tf_record.py --data_dir=../MusicObjectDetector/data/training_validation_test_without_stafflines --set=training --annotations_dir=Annotations --output_path=../MusicObjectDetector/data/training_validation_test_without_stafflines/training.record --label_map_path=../MusicObjectDetector/mapping_reduced_class2.txt)
(cd ../research; python ../MusicObjectDetector/create_muscima_tf_record.py --data_dir=../MusicObjectDetector/data/training_validation_test_without_stafflines --set=validation --annotations_dir=Annotations --output_path=../MusicObjectDetector/data/training_validation_test_without_stafflines/validation.record --label_map_path=../MusicObjectDetector/mapping_reduced_class2.txt)
(cd ../research; python ../MusicObjectDetector/create_muscima_tf_record.py --data_dir=../MusicObjectDetector/data/training_validation_test_without_stafflines --set=test --annotations_dir=Annotations --output_path=../MusicObjectDetector/data/training_validation_test_without_stafflines/test.record --label_map_path=../MusicObjectDetector/mapping_reduced_class2.txt)
```
    
These two scripts will download the datasets automatically, generate cropped images along an Annotation.txt file and split the images into three reproducible parts for training, validation and test. 

Images will be cropped first vertically along the staffs and then horizontally (red boxes) like this (with orange regions overlapping between two regions):
![Cropping of images](images/w-05_p006_crop_regions.png) 

# License

Published under MIT License,

Copyright (c) 2017 [Alexander Pacha](http://alexanderpacha.com), [TU Wien](https://www.ims.tuwien.ac.at/people/alexander-pacha) and Kwon-Young Choi

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
