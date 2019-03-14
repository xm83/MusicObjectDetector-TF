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

echo "Validate with $($configuration) configuration"

# Legacy slim-based evaluation
Start-Transcript -path "$($pathToTranscript)/Validate-$($configuration).txt" -append
python research/object_detection/legacy/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToData)/$($configuration)" --eval_dir="$($pathToData)/$($configuration)/eval"
Stop-Transcript
