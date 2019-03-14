$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF"
$pathToSourceRoot = "$($pathToGitRoot)/MusicObjectDetector"
$pathToTranscript = "$($pathToSourceRoot)/Transcripts"
$pathToData = "$($pathToSourceRoot)/data"
cd $pathToGitRoot

echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToGitRoot)/research;$($pathToGitRoot)/research/slim;$($pathToSourceRoot)"

################################################################
# Available configurations - uncomment the one to actually run #
################################################################
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_muscima_3"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_muscima_dim_clustering"
$configuration = "ssd_inception_v2_focal_loss_muscima"

echo "Training with $($configuration) configuration"

# Legacy slim-based
Start-Transcript -path "$($pathToTranscript)/Train-$($configuration).txt" -append
python research/object_detection/legacy/train.py --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --train_dir="$($pathToData)/$($configuration)"
Stop-Transcript

# # Estimator-based
# Start-Transcript -path "$($pathToTranscript)/TrainEval-$($configuration).txt" -append
# python research/object_detection/model_main.py --alsologtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --model_dir="$($pathToData)/$($configuration)"
# Stop-Transcript