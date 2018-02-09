$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF"
$pathToSourceRoot = "$($pathToGitRoot)/MusicObjectDetector"
$pathToTranscript = "$($pathToSourceRoot)/Transcripts"
$pathToData = "$($pathToSourceRoot)/data"
#$pathToData = "\\MONSTI\MusicObjectDetector-TF_Results"
cd $pathToGitRoot/research

################################################################
# Available configurations - uncomment the one to actually run #
################################################################
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering2"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering2_rms"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering2_rms_without_stafflines"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering3"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering3_rms"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering4"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering4_rms"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_reduced_classes_dim_clust2_rms"
# $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_reduced_classes_dim_clust2_rms_without_stafflines"
## $configuration = "faster_rcnn_inception_resnet_v2_atrous_pretrained_rms"

# $configuration = "faster_rcnn_resnet50_pretrained_dim_clust2_rms"
$configuration = "faster_rcnn_resnet50_pretrained_dim_clust2_rms_without_stafflines"
## $configuration = "faster_rcnn_resnet50_pretrained_reduced_classes_dim_clust2_rms"
## $configuration = "faster_rcnn_resnet50_pretrained_reduced_classes_dim_clust2_rms_without_stafflines"

## $configuration = "rfcn_inception_resnet_v2_atrous_pretrained_dim_clust2_rms"
## $configuration = "rfcn_inception_resnet_v2_atrous_pretrained_dim_clust2_rms_without_stafflines"
# $configuration = "rfcn_resnet50_pretrained_dim_clust2_rms"
# $configuration = "rfcn_resnet50_pretrained_dim_clust2_rms_without_stafflines"

## $configuration = "ssd_inception_v2_300x600_pretrained_dim_clust2_rms"
## $configuration = "ssd_inception_v2_300x600_pretrained_dim_clust2_rms_without_stafflines"
## $configuration = "ssd_mobilenet_v1_300x600_pretrained_dim_clust2_rms"
## $configuration = "ssd_mobilenet_v1_300x600_pretrained_dim_clust2_rms_without_stafflines"


Start-Transcript -path "$($pathToTranscript)/TrainModel-$($configuration).txt" -append
echo "Training with $($configuration) configuration"
python object_detection/train.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --train_dir="$($pathToData)/checkpoints-$($configuration)-train"
Stop-Transcript
