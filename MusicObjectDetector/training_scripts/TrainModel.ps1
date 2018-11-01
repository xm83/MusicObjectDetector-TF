$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF"
$pathToSourceRoot = "$($pathToGitRoot)/MusicObjectDetector"
$pathToTranscript = "$($pathToSourceRoot)/Transcripts"
$pathToData = "$($pathToSourceRoot)/data"
#$pathToData = "\\MONSTI\MusicObjectDetector-TF_Results"
cd $pathToGitRoot

#echo "Appending required paths to temporary PYTHONPATH"
#$env:PYTHONPATH = "$($pathToGitRoot);$($pathToGitRoot)/research;$($pathToGitRoot)/research/slim;$($pathToSourceRoot)"

################################################################
# Available configurations - uncomment the one to actually run #
################################################################
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_muscima_dimension_clustering_rms"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_muscima_dimension_clustering_rms_1200_proposals"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_muscima_dimension_clustering_rms_2000_proposals"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_deepscores_1"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_deepscores_2"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_mensural_1"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_mensural_2"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_mensural_3"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_muscima_1"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_muscima_3"
#$configuration = "faster_rcnn_inc_resnet_v2_muscima_1"
#$configuration = "ssd_resnet50_retinanet_muscima_1"
#$configuration = "ssd_resnet50_retinanet_muscima_2"
#$configuration = "ssd_resnet50_retinanet_muscima_3"
#$configuration = "ssd_resnet50_retinanet_muscima_4"
#$configuration = "ssd_resnet50_retinanet_muscima_5"
#$configuration = "ssd_resnet50_retinanet_muscima_6"
#$configuration = "ssd_resnet50_retinanet_muscima_7"
#$configuration = "ssd_inception_v2_focal_loss_muscima_1"
#$configuration = "ssd_inception_v2_focal_loss_muscima_2"
#$configuration = "ssdlite_mobilenet_v2_muscima_1"
#$configuration = "ssdlite_mobilenet_v2_muscima_2"
#$configuration = "ssdlite_mobilenet_v2_muscima_3"
#$configuration = "faster_rcnn_inception_v2_pretrained_muscima_1"
#$configuration = "faster_rcnn_inception_v2_pretrained_muscima_2"
#$configuration = "faster_rcnn_inception_v2_pretrained_muscima_3"
#$configuration = "faster_rcnn_inception_v2_pretrained_muscima_3_no_mirror"
#$configuration = "faster_rcnn_inception_v2_pretrained_muscima_4"
#$configuration = "faster_rcnn_inception_v2_pretrained_muscima_5"
#$configuration = "faster_rcnn_inception_v2_pretrained_muscima_6"
#$configuration = "faster_rcnn_inception_v2_pretrained_muscima_7"
#$configuration = "faster_rcnn_inception_v2_muscima_1"

echo "Training with $($configuration) configuration"

# Legacy slim-based
Start-Transcript -path "$($pathToTranscript)/Train-$($configuration).txt" -append
python research/object_detection/legacy/train.py --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --train_dir="$($pathToData)/checkpoints-$($configuration)-train"
Stop-Transcript

# # Estimator-based
# Start-Transcript -path "$($pathToTranscript)/TrainEval-$($configuration).txt" -append
# python research/object_detection/model_main.py --alsologtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --model_dir="$($pathToData)/$($configuration)"
# Stop-Transcript


# C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\training_scripts\ValidateModel.ps1