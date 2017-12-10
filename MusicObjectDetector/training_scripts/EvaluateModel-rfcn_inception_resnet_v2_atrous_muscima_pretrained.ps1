$pathToGitRoot = "C:/Users/alpa/Repositories/MusicObjectDetector-TF"
$pathToSourceRoot = "$($pathToGitRoot)/MusicObjectDetector"
$pathToTranscript = "$($pathToSourceRoot)/Transcripts"
$configuration = "rfcn_inception_resnet_v2_atrous_muscima_pretrained"

cd $pathToGitRoot/research

Start-Transcript -path "$($pathToTranscript)/EvaluateModel-$($configuration).txt" -append
# echo "Validate with $($configuration) configuration"
# python object_detection/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToSourceRoot)/data/training-checkpoints-$($configuration)" --eval_dir="$($pathToSourceRoot)/data/validation-checkpoints-$($configuration)"

echo "Testing with $($configuration) configuration"
python object_detection/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToSourceRoot)/data/training-checkpoints-$($configuration)" --eval_dir="$($pathToSourceRoot)/data/test-checkpoints-$($configuration)"
python object_detection/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToSourceRoot)/data/training-checkpoints-$($configuration)" --eval_dir="$($pathToSourceRoot)/data/test-weighted-checkpoints-$($configuration)"
Stop-Transcript

# Continue the training
# C:\Users\alpa\Repositories\MusicObjectDetector-TF\MusicObjectDetector\training_scripts\TrainModel-rfcn_inception_resnet_v2_atrous_muscima_pretrained.ps1