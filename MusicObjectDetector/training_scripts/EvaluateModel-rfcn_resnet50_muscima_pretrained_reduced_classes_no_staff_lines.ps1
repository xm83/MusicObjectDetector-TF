$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF"
$pathToSourceRoot = "$($pathToGitRoot)/MusicObjectDetector"
$pathToTranscript = "$($pathToSourceRoot)/Transcripts"
$configuration = "rfcn_resnet50_muscima_pretrained_reduced_classes_no_staff_lines"

cd $pathToGitRoot/research

Start-Transcript -path "$($pathToTranscript)/EvaluateModel-$($configuration).txt" -append
echo "Validate with $($configuration) configuration"
# Validation
python object_detection/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToSourceRoot)/data/training-checkpoints-$($configuration)" --eval_dir="$($pathToSourceRoot)/data/validation-checkpoints-$($configuration)"
# Test
# python object_detection/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToSourceRoot)/data/training-checkpoints-$($configuration)" --eval_dir="$($pathToSourceRoot)/data/test-checkpoints-$($configuration)"
# python object_detection/eval.py --logtostderr --pipeline_config_path="$($pathToSourceRoot)/configurations/$($configuration).config" --checkpoint_dir="$($pathToSourceRoot)/data/training-checkpoints-$($configuration)" --eval_dir="$($pathToSourceRoot)/data/test-weighted-checkpoints-$($configuration)"
Stop-Transcript

# Continue the training
C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\training_scripts\TrainModel-$configuration.ps1

