# Running this demo

Download the trained model from 
- [2018-07-30_faster-rcnn_inception-resnet-v2_full-page_muscima.pb](https://github.com/apacha/MusicObjectDetector-TF/releases/download/full-page-detection-v2/2018-07-30_faster-rcnn_inception-resnet-v2_full-page_muscima.pb)

and put it into the demo directory.

Then run `standalone_inference_over_image.py` from inside the demo directory:

```bash
python standalone_inference_over_image.py \
    --detection_inference_graph 2018-07-30_faster-rcnn_inception-resnet-v2_full-page_muscima.pb \
    --input_image w-21_p008.png \
    --detection_label_map category_mapping.txt \
    --output_image annotated_image.jpg \
    --output_result output_transcript.txt
```

Note that the depicted parameters are the default-parameter, so if you are happy with them, they can be omitted (e.g. the label_maps).
