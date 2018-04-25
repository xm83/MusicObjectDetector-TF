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
$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering_rms"


Start-Transcript -path "$($pathToTranscript)/ValidateModel-$($configuration).txt" -append
echo "Validate with $($configuration) configuration"
python research/object_detection/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToData)/checkpoints-$($configuration)-train" --eval_dir="$($pathToData)/checkpoints-$($configuration)-validate"
Stop-Transcript

