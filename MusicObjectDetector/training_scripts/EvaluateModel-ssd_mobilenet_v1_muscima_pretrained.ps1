$pathToGitRoot = "C:/Users/alpa/Repositories/MusicObjectDetector-TF/"
$pathToSourceRoot = "C:/Users/alpa/Repositories/MusicObjectDetector-TF/MusicObjectDetector/"
$pathToTranscript = "$($pathToSourceRoot)"

# Allowing wider outputs https://stackoverflow.com/questions/7158142/prevent-powergui-from-truncating-the-output
$pshost = get-host
$pswindow = $pshost.ui.rawui
$newsize = $pswindow.buffersize
$newsize.height = 9999
$newsize.width = 1500
$pswindow.buffersize = $newsize


cd C:/Users/alpa/Repositories/MusicObjectDetector-TF/research

Start-Transcript -path "$($pathToTranscript)Evaluate-ssd_mobilenet_v1_muscima configuration_pretrained.txt" -append
echo "Validate with ssd_mobilenet_v1_muscima configuration_150x300"
python object_detection/eval.py --logtostderr --pipeline_config_path=C:\Users\alpa\Repositories\MusicObjectDetector-TF\MusicObjectDetector\configurations\ssd_mobilenet_v1_muscima configuration_pretrained.config --checkpoint_dir=C:\Users\alpa\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\training-checkpoints-ssd_mobilenet_v1_muscima configuration_pretrained --eval_dir=C:\Users\alpa\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\validation-checkpoints-ssd_mobilenet_v1_muscima configuration_pretrained
Stop-Transcript
