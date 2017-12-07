$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF"
$pathToSourceRoot = "$($pathToGitRoot)/MusicObjectDetector"
$pathToTranscript = "$($pathToSourceRoot)/Transcripts"
$configuration = "faster_rcnn_nas_muscima_pretrained"

cd $pathToGitRoot/research

Start-Transcript -path "$($pathToTranscript)/TrainModel-$($configuration).txt" -append
echo "Training with $($configuration) configuration"
python object_detection/train.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --train_dir="$($pathToSourceRoot)/data/training-checkpoints-$($configuration)"
Stop-Transcript
