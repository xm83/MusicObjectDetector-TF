import argparse
import os

import pandas as pd


def generate_mapping(args):
    annotations = pd.read_csv(os.path.join(args.normalized_dataset_dir, "annotations.csv"))

    class_names = annotations["class_name"].unique()
    class_names.sort()

    with open(args.mapping_output_path, "w") as f:
        for i, class_name in enumerate(class_names):
            f.write("""
item{{
    id: {}
    name: '{}'
}}
""".format(i + 1, class_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized_dataset_dir", default="./data/normalized/deepscores")
    parser.add_argument("--mapping_output_path", default="./data/normalized/deepscores/mapping.txt")
    args = parser.parse_args()

    generate_mapping(args)
