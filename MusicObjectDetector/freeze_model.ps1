$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF/"
$pathToSourceRoot = "$($pathToGitRoot)/MusicObjectDetector"
$pathToTranscript = "$($pathToSourceRoot)/transcripts"
cd $pathToGitRoot

Start-Transcript -path "$($pathToTranscript)/Freeze_Model.txt" -append

# Make sure that python finds all modules inside this directory
echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToGitRoot)/research;$($pathToGitRoot)/research/slim;$($pathToSourceRoot)"

# Replace with your path to the checkpoint folder and the respective checkpoint number
$pathToCheckpoint = "C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\checkpoints-faster_rcnn_inception_resnet_v2_atrous_pretrained_dimension_clustering_rms_2000_proposals-train"
$checkpointNumber = "141270"

python research/object_detection/export_inference_graph.py `
    --input_type image_tensor `
    --pipeline_config_path "$($pathToCheckpoint)\pipeline.config"  `
    --trained_checkpoint_prefix "$($pathToCheckpoint)\model.ckpt-$($checkpointNumber)" `
    --output_directory output_inference_graph
	
Stop-Transcript
