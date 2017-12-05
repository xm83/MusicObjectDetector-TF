$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF/"
$pathToSourceRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF/MusicObjectDetector/"
$pathToTranscript = "$($pathToSourceRoot)"

cd $pathToSourceRoot/../research

Start-Transcript -path "$($pathToTranscript)Transcript-faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained.txt" -append

echo "Train with faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained configuration"
python object_detection/train.py --logtostderr --pipeline_config_path=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\configurations\faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained.config --train_dir=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\training-checkpoints-faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained

Stop-Transcript
