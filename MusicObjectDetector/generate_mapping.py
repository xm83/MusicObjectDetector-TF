import os
import muscima.io
import glob
from collections import Counter
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir")
parser.add_argument("mapping_filepath")
parser.add_argument("--nbr_example_limit", type=int, default=50)
parser.add_argument("--remove_line_shaped_or_construct", type=bool, default=False)
args = parser.parse_args()

names = glob.glob(os.path.join(args.dataset_dir, "*.xml"))
data = {}
for name in tqdm(names):
    data[name] = muscima.io.parse_cropobject_list(name)
datas = []

for value in data.values():
    for val in value:
        datas.append(val)

c = Counter([x.clsname for x in datas])
__import__('pdb').set_trace()
# keys = [key for key, _ in c.most_common()]
# df = pandas.DataFrame.from_dict(c, orient='index')
# ax = df.ix[keys].plot(kind='bar', figsize=(20, 20))
# ax.get_figure().savefig("class_histogram.png")

exception = ["double_sharp",
             "numeral_2",
             "numeral_5",
             "numeral_6",
             "numeral_7",
             "numeral_8"]
line_shaped_or_construct = ["stem",
                            "beam",
                            "thin_barline",
                            "measure_separator",
                            "slur",
                            "tie",
                            "key_signature",
                            "dynamics_text",
                            "hairpin-decr.",
                            "other_text",
                            "tuple",
                            "hairpin-cresc.",
                            "time_signature",
                            "staff_grouping",
                            "trill",
                            "tenuto",
                            "tempo_text",
                            "multi-staff_brace",
                            "multiple-note_tremolo",
                            "repeat",
                            "multi-staff_bracket",
                            "tuple_bracket/line"]
filtered_class_id = []
for key, value in c.items():
    if value >= args.nbr_example_limit or key in exception:
        if args.remove_line_shaped_or_construct:
            if key not in line_shaped_or_construct:
                filtered_class_id.append(key)
        else:
            filtered_class_id.append(key)
filtered_class_id.sort()
with open(args.mapping_filepath, "w") as f:
    for i, classname in enumerate(filtered_class_id):
        f.write("""
item{{
  id: {}
  name: '{}'
}}
""".format(i + 1, classname))
