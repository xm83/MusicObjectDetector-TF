$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF"
$pathToSourceRoot = "$($pathToGitRoot)/MusicObjectDetector"
$pathToTranscript = "$($pathToSourceRoot)/Transcripts"
cd $pathToGitRoot/research

################################################################
# Available configurations - uncomment the one to actually run #
################################################################
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_no_staff_lines"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_reduced_classes"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_reduced_classes_no_staff_lines"
# $configuration = "faster_rcnn_resnet50_muscima_pretrained"
# $configuration = "faster_rcnn_resnet50_muscima_pretrained2"
# $configuration = "faster_rcnn_resnet50_muscima_windows"
# $configuration = "faster_rcnn_resnet50_muscima_windows_2"
$configuration = "faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_with_stafflines_more_scales_and_ratios"
# $configuration = "rfcn_inception_resnet_v2_atrous_muscima_pretrained"
# $configuration = "rfcn_inception_resnet_v2_atrous_muscima_pretrained_reduced_classes"
# $configuration = "rfcn_resnet50_muscima"
# $configuration = "rfcn_resnet50_muscima_pretrained_no_staff_lines"
# $configuration = "rfcn_resnet50_muscima_pretrained_reduced_classes"
# $configuration = "rfcn_resnet50_muscima_pretrained_reduced_classes_no_staff_lines"
# $configuration = "rfcn_resnet50_muscima_reduced_classes"
# $configuration = "rfcn_resnet50_muscima_reduced_classes_no_staff_lines"
# $configuration = "ssd_inception_v2_muscima_150x300_pretrained"
# $configuration = "ssd_inception_v2_muscima_150x300_pretrained_reduced_classes"
# $configuration = "ssd_inception_v2_muscima_150x300_pretrained_reduced_classes_no_stafflines"
# $configuration = "ssd_mobilenet_v1_muscima_150x300"
# $configuration = "ssd_mobilenet_v1_muscima_150x300_pretrained"


Start-Transcript -path "$($pathToTranscript)/EvaluateModel-$($configuration).txt" -append
echo "Validate with $($configuration) configuration"
python object_detection/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToSourceRoot)/data/training-checkpoints-$($configuration)" --eval_dir="$($pathToSourceRoot)/data/validation-checkpoints-$($configuration)"
Stop-Transcript

# Continue the training
# C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\training_scripts\TrainModel-rfcn_resnet50_muscima_reduced_classes.ps1

