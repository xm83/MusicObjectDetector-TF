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

# echo "Testing correct setup"
# cd ../research
# python object_detection/builders/model_builder_test.py
# 
# echo "Generating data-record in Tensorflow-format"
# cd ../MusicObjectDetector
# python muscima_image_cutter.py
# python DatasetSplitter.py
# python create_muscima_tf_record.py --data_dir=data --set=training --annotations_dir=Annotations --output_path=training.record --label_map_path=mapping.txt
# python create_muscima_tf_record.py --data_dir=data --set=validation --annotations_dir=Annotations --output_path=validation.record --label_map_path=mapping.txt
# python create_muscima_tf_record.py --data_dir=data --set=test --annotations_dir=Annotations --output_path=test.record --label_map_path=mapping.txt

$pathToPipelineConfig = "$($pathToSourceRoot)faster_rcnn_resnet50_muscima_windows.config"
$trainingDirectory = $pathToSourceRoot
cd ../research
#python object_detection/train.py --logtostderr --pipeline_config_path=$pathToPipelineConfig --train_dir=$trainingDirectory
python object_detection/train.py --logtostderr --pipeline_config_path=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\faster_rcnn_resnet50_muscima_windows.config --train_dir=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\training-checkpoints

python object_detection/eval.py --logtostderr --pipeline_config_path=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\faster_rcnn_resnet50_muscima_windows.config --checkpoint_dir=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data --eval_dir=C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\data\validation-checkpoints


Stop-Transcript
