$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF2"
$pathToSourceRoot = "$($pathToGitRoot)/MusicObjectDetector"
$pathToTranscript = "$($pathToSourceRoot)/Transcripts"
$pathToData = "$($pathToSourceRoot)/data"
#$pathToData = "\\MONSTI\MusicObjectDetector-TF_Results"
cd $pathToGitRoot

#echo "Appending required paths to temporary PYTHONPATH"
#$env:PYTHONPATH = "$($pathToGitRoot);$($pathToGitRoot)/research;$($pathToSourceRoot)"

python research/object_detection/model_main.py --pipeline_config_path="C:/Users/Alex/Repositories/MusicObjectDetector-TF2/research/object_detection/samples/configs/ssd_mobilenet_v1_0.75_depth_quantized_300x300_pets_sync.config" --model_dir="$($pathToData)/pet-train" --alsologtostderr


exit

################################################################
# Available configurations - uncomment the one to actually run #
################################################################
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering_rms"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering_rms_1200_proposals"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering_rms_2000_proposals"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_deepscores_1"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_mensural_1"
#$configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_muscima_1"
$configuration = "faster_rcnn_inc_resnet_v2_muscima_1"
$configuration = "faster_rcnn_resnet50_muscima"

Start-Transcript -path "$($pathToTranscript)/TrainModel-$($configuration).txt" -append
echo "Training with $($configuration) configuration"
python research/object_detection/model_main.py --alsologtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --model_dir="$($pathToData)/checkpoints-$($configuration)"
Stop-Transcript
