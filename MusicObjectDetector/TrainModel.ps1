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

Start-Transcript -path "$($pathToTranscript)Transcript.txt" -append

cd $pathToSourceRoot
echo "Appending research folder $($pathToGitRoot)research to temporary PYTHONPATH"
$env:PYTHONPATH = $env:PYTHONPATH;"$($pathToSourceRoot)research"

$pathToPipelineConfig = "$($pathToSourceRoot)faster_rcnn_resnet50_muscima_windows.config"
$trainingDirectory = $pathToSourceRoot
cd ../research
#python object_detection/train.py --logtostderr --pipeline_config_path=$pathToPipelineConfig --train_dir=$trainingDirectory

echo "Train with resnet50-muscima-window configuration"
python object_detection/train.py --logtostderr --pipeline_config_path=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\configurations\faster_rcnn_resnet50_muscima_windows.config --train_dir=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\training-checkpoints-faster_rcnn_resnet50_muscima_windows
# Validate with resnet50-muscima-window configuration
python object_detection/eval.py --logtostderr --pipeline_config_path=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\configurations\faster_rcnn_resnet50_muscima_windows.config --checkpoint_dir=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\training-checkpoints-faster_rcnn_resnet50_muscima_windows --eval_dir=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\validation-checkpoints-faster_rcnn_resnet50_muscima_windows


exit 

echo "Train with resnet50-muscima-window-2 configuration"
python object_detection/train.py --logtostderr --pipeline_config_path=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\configurations\faster_rcnn_resnet50_muscima_windows_2.config --train_dir=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\training-checkpoints-faster_rcnn_resnet50_muscima_windows_2
# Validate with resnet50-muscima-window-2 configuration
python object_detection/eval.py --logtostderr --pipeline_config_path=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\configurations\faster_rcnn_resnet50_muscima_windows_2.config --checkpoint_dir=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\training-checkpoints-faster_rcnn_resnet50_muscima_windows_2 --eval_dir=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\validation-checkpoints-faster_rcnn_resnet50_muscima_windows_2

Stop-Transcript
