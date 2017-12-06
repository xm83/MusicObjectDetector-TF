$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF"
$pathToSourceRoot = "$($pathToGitRoot)/MusicObjectDetector"
$pathToTranscript = "$($pathToSourceRoot)/Transcripts"
$configuration = "ssd_inception_v2_muscima_150x300_pretrained_reduced_classes"

cd $pathToGitRoot/research

Start-Transcript -path "$($pathToTranscript)/TrainModel-$($configuration).txt" -append
echo "Training with $($configuration) configuration"
python object_detection/train.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --train_dir="$($pathToSourceRoot)/data/training-checkpoints-$($configuration)"
Stop-Transcript
