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

echo "Testing correct setup"
cd ../research
python object_detection/builders/model_builder_test.py

echo "Generating data-record in Tensorflow-format"
cd ../MusicObjectDetector
python muscima_image_cutter.py

python DatasetSplitter.py --source_directory=data/muscima_pp_cropped_images_with_stafflines --destination_directory=data/training_validation_test_with_stafflines
python DatasetSplitter.py --source_directory=data/muscima_pp_cropped_images_without_stafflines --destination_directory=data/training_validation_test_without_stafflines

python create_muscima_tf_record.py --data_dir=data/training_validation_test_with_stafflines --set=training --annotations_dir=Annotations --output_path=data/all_classes_with_staff_lines_writer_independent_split/training.record --label_map_path=mapping_all_classes.txt
python create_muscima_tf_record.py --data_dir=data/training_validation_test_with_stafflines --set=validation --annotations_dir=Annotations --output_path=data/all_classes_with_staff_lines_writer_independent_split/validation.record --label_map_path=mapping_all_classes.txt
python create_muscima_tf_record.py --data_dir=data/training_validation_test_with_stafflines --set=test --annotations_dir=Annotations --output_path=data/all_classes_with_staff_lines_writer_independent_split/test.record --label_map_path=mapping_all_classes.txt

python create_muscima_tf_record.py --data_dir=data/training_validation_test_without_stafflines --set=training --annotations_dir=Annotations --output_path=data/all_classes_without_staff_lines_writer_independent_split/training.record --label_map_path=mapping_all_classes.txt
python create_muscima_tf_record.py --data_dir=data/training_validation_test_without_stafflines --set=validation --annotations_dir=Annotations --output_path=data/all_classes_without_staff_lines_writer_independent_split/validation.record --label_map_path=mapping_all_classes.txt
python create_muscima_tf_record.py --data_dir=data/training_validation_test_without_stafflines --set=test --annotations_dir=Annotations --output_path=data/all_classes_without_staff_lines_writer_independent_split/test.record --label_map_path=mapping_all_classes.txt
                                                                                                                                                      
 python create_muscima_tf_record.py --data_dir=data/training_validation_test_with_stafflines --set=training --annotations_dir=Annotations --output_path=data/71_classes_with_staff_lines_writer_independent_split/training.record --label_map_path=mapping_71_classes.txt
 python create_muscima_tf_record.py --data_dir=data/training_validation_test_with_stafflines --set=validation --annotations_dir=Annotations --output_path=data/71_classes_with_staff_lines_writer_independent_split/validation.record --label_map_path=mapping_71_classes.txt
 python create_muscima_tf_record.py --data_dir=data/training_validation_test_with_stafflines --set=test --annotations_dir=Annotations --output_path=data/71_classes_with_staff_lines_writer_independent_split/test.record --label_map_path=mapping_71_classes.txt
 
 python create_muscima_tf_record.py --data_dir=data/training_validation_test_without_stafflines --set=training --annotations_dir=Annotations --output_path=data/71_classes_without_staff_lines_writer_independent_split/training.record --label_map_path=mapping_71_classes.txt
 python create_muscima_tf_record.py --data_dir=data/training_validation_test_without_stafflines --set=validation --annotations_dir=Annotations --output_path=data/71_classes_without_staff_lines_writer_independent_split/validation.record --label_map_path=mapping_71_classes.txt
 python create_muscima_tf_record.py --data_dir=data/training_validation_test_without_stafflines --set=test --annotations_dir=Annotations --output_path=data/71_classes_without_staff_lines_writer_independent_split/test.record --label_map_path=mapping_71_classes.txt

Stop-Transcript
