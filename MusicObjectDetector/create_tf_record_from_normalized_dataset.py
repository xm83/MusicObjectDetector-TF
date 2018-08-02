import hashlib
import io
import os

import PIL.Image
import tensorflow as tf
from typing import List, Dict

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from tqdm import tqdm
import pandas as pd

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'training', 'Convert training set, validation set or '
                                       'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

SETS = ['training', 'validation', 'test']


def annotations_to_tf_example_list(annotations: pd.DataFrame,
                                   dataset_directory: str,
                                   label_map_dict: Dict[str, int]) -> List[tf.train.Example]:
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      example: single row of the pandas dataframe of the CSV files
      dataset_directory: Path to root directory holding of the normalized dataset
      label_map_dict: A map from string label names to integers ids.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """

    examples = []
    total_number_of_images = len(annotations["path_to_image"].unique())
    for path_to_image, image_annotations in tqdm(annotations.groupby("path_to_image"),
                                                 desc="Serializing annotations",
                                                 total=total_number_of_images):

        full_path = os.path.join(dataset_directory, path_to_image)
        with tf.gfile.GFile(full_path, 'rb') as fid:
            encoded_png = fid.read()
        encoded_png_io = io.BytesIO(encoded_png)
        image = PIL.Image.open(encoded_png_io)
        if image.format != 'PNG':
            raise ValueError('Image format not PNG')
        key = hashlib.sha256(encoded_png).hexdigest()

        width = image.width
        height = image.height

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []

        for index, example in image_annotations.iterrows():
            xmin.append(float(example['left']) / width)
            ymin.append(float(example['top']) / height)
            xmax.append(float(example['right']) / width)
            ymax.append(float(example['bottom']) / height)
            classes.append(label_map_dict[example['class_name']])
            classes_text.append(example['class_name'].encode('utf8'))

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(
                path_to_image.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(
                path_to_image.encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_png),
            'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
        }))

        examples.append(example)

    return examples


def main(_):
    dataset_split = FLAGS.set
    if dataset_split not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    data_dir = FLAGS.data_dir

    os.makedirs(os.path.dirname(FLAGS.output_path), exist_ok=True)

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    annotations_path = os.path.join(data_dir, dataset_split + '.csv')

    annotations = pd.read_csv(annotations_path)
    tf_examples = annotations_to_tf_example_list(annotations, data_dir, label_map_dict)

    for tf_example in tqdm(tf_examples,
                           desc="Serializing annotations from {0} set into TF-Example".format(dataset_split)):
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
