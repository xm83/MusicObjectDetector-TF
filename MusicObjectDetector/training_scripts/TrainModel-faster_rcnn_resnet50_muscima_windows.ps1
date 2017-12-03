$pathToSourceRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF/MusicObjectDetector/"
$pathToTranscript = "$($pathToSourceRoot)"

# Allowing wider outputs https://stackoverflow.com/questions/7158142/prevent-powergui-from-truncating-the-output
$pshost = get-host
$pswindow = $pshost.ui.rawui
$newsize = $pswindow.buffersize
$newsize.height = 9999
$newsize.width = 1500
$pswindow.buffersize = $newsize


cd C:/Users/Alex/Repositories/MusicObjectDetector-TF/research

Start-Transcript -path "$($pathToTranscript)TrainTranscript-faster_rcnn_resnet50_muscima_windows.txt" -append

echo "Train with resnet50-muscima-window configuration"
python object_detection/train.py --logtostderr --pipeline_config_path=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\configurations\faster_rcnn_resnet50_muscima_windows.config --train_dir=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\training-checkpoints-faster_rcnn_resnet50_muscima_windows

Stop-Transcript
