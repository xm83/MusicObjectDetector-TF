
# Training on MUSCIMA++ dataset

## Dataset generation

* Download MUSCIMA++ dataset and MUSCIMA images
* Generate Pascal VOC xml annotations
* Generate label mapping mapping.txt
* Generate ImageSets train.txt, validation.txt, test.txt
  Contains basename of images grouped following train, validation and test set
* Convert to TFRecord format using create_muscima_tf_record.py script for training, validation and test set

```
python create_muscima_tf_record.py \
    --data_dir=MusicObjectDetector/data \
    --set=validation \
    --annotations_dir=MusicObjectDetector/data/Annotations \
    --output_path=MusicObjectDetector/data/validation.record \
    --label_map_path=MusicObjectDetector/mapping.txt
```

## Training

Produce a configuration file: samples/configs/faster_rcnn_resnet50_muscima.config

Configure:

* nbr of classes
* dataset path
* ...


