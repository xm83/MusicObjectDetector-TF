$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF"
$pathToSourceRoot = "$($pathToGitRoot)/MusicObjectDetector"
$pathToTranscript = "$($pathToSourceRoot)/Transcripts"
$pathToData = "$($pathToSourceRoot)/data"
#$pathToData = "\\MONSTI\MusicObjectDetector-TF_Results"
cd $pathToGitRoot

echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToGitRoot)/research;$($pathToGitRoot)/research/slim;$($pathToSourceRoot)"

################################################################
# Available configurations - uncomment the one to actually run #
################################################################
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering_rms"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering_rms_1200_proposals"
$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering_rms_2000_proposals"


Start-Transcript -path "$($pathToTranscript)/TrainModel-$($configuration).txt" -append
echo "Training with $($configuration) configuration"
python research/object_detection/train.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --train_dir="$($pathToData)/checkpoints-$($configuration)-train"
Stop-Transcript
