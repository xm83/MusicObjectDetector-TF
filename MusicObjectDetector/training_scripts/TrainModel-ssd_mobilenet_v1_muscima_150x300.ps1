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

cd $pathToSourceRoot
echo "Appending research folder $($pathToGitRoot)research to temporary PYTHONPATH"
$env:PYTHONPATH = $env:PYTHONPATH;"$($pathToSourceRoot)research"

$pathToPipelineConfig = "$($pathToSourceRoot)faster_rcnn_resnet50_muscima_windows.config"
$trainingDirectory = $pathToSourceRoot
cd ../research
#python object_detection/train.py --logtostderr --pipeline_config_path=$pathToPipelineConfig --train_dir=$trainingDirectory

Start-Transcript -path "$($pathToTranscript)MobileNet2Transcript.txt" -append
echo "Train with ssd_mobilenet_v1_muscima configuration_150x300"
python object_detection/train.py --logtostderr --pipeline_config_path=C:\Users\alpa\Repositories\MusicObjectDetector-TF\MusicObjectDetector\configurations\ssd_mobilenet_v1_muscima_150x300.config --train_dir=C:\Users\alpa\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\training-checkpoints-ssd_mobilenet_v1_muscima_150x300
Stop-Transcript