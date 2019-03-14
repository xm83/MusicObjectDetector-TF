$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF"
$pathToSourceRoot = "$($pathToGitRoot)/MusicObjectDetector"
$pathToTranscript = "$($pathToSourceRoot)/Transcripts"
$pathToData = "$($pathToSourceRoot)/data"
cd $pathToGitRoot

echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToGitRoot)/research;$($pathToGitRoot)/research/slim;$($pathToSourceRoot)"

#####################################################
# Available configurations, see the other scripts   #
#####################################################
$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_muscima_3"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_muscima_dim_clustering"
# $configuration = "ssd_inception_v2_focal_loss_muscima"

Start-Transcript -path "$($pathToTranscript)/Test-$($configuration).txt" -append
echo "Testing with $($configuration) configuration"
python research/object_detection/legacy/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToData)/$($configuration)" --eval_dir="$($pathToData)/$($configuration)/test" --write_csv
Stop-Transcript
