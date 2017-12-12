$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF"
$pathToSourceRoot = "$($pathToGitRoot)/MusicObjectDetector"
$pathToTranscript = "$($pathToSourceRoot)/Transcripts"
$configuration = "faster_rcnn_resnet50_muscima_pretrained2"

cd $pathToGitRoot/research

Start-Transcript -path "$($pathToTranscript)/EvaluateModel-$($configuration).txt" -append
# echo "Validate with $($configuration) configuration"
# python object_detection/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToSourceRoot)/data/training-checkpoints-$($configuration)" --eval_dir="$($pathToSourceRoot)/data/validation-checkpoints-$($configuration)"

echo "Testing with $($configuration) configuration"
python object_detection/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToSourceRoot)/data/training-checkpoints-$($configuration)" --eval_dir="$($pathToSourceRoot)/data/test-checkpoints-$($configuration)"
python object_detection/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToSourceRoot)/data/training-checkpoints-$($configuration)" --eval_dir="$($pathToSourceRoot)/data/test-weighted-checkpoints-$($configuration)"
Stop-Transcript

