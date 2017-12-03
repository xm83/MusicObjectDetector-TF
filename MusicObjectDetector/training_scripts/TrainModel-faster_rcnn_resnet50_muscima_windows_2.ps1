$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF/"
$pathToSourceRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF/MusicObjectDetector/"
$pathToTranscript = "$($pathToSourceRoot)"

# Allowing wider outputs https://stackoverflow.com/questions/7158142/prevent-powergui-from-truncating-the-output
$pshost = get-host
$pswindow = $pshost.ui.rawui
$newsize = $pswindow.buffersize
$newsize.height = 9999
$newsize.width = 1500
$pswindow.buffersize = $newsize


cd $pathToSourceRoot/../research

Start-Transcript -path "$($pathToTranscript)Transcript.txt" -append

echo "Train with faster_rcnn_resnet50_muscima_windows_2 configuration"
python object_detection/train.py --logtostderr --pipeline_config_path=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\configurations\faster_rcnn_resnet50_muscima_windows_2.config --train_dir=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\training-checkpoints-faster_rcnn_resnet50_muscima_windows_2

Stop-Transcript
