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

python create_muscima_tf_record.py --data_dir=data\training_validation_test_with_stafflines  	--set=training 		--annotations_dir=Annotations 	--output_path=data\training_with_stafflines_all_classes.record 			--label_map_path=mapping.txt
python create_muscima_tf_record.py --data_dir=data\training_validation_test_with_stafflines  	--set=validation 	--annotations_dir=Annotations 	--output_path=data\validation_with_stafflines_all_classes.record 		--label_map_path=mapping.txt
python create_muscima_tf_record.py --data_dir=data\training_validation_test_with_stafflines  	--set=test 			--annotations_dir=Annotations 	--output_path=data\test_with_stafflines_all_classes.record 				--label_map_path=mapping.txt
                                                                           
python create_muscima_tf_record.py --data_dir=data\training_validation_test_without_stafflines  --set=training 		--annotations_dir=Annotations 	--output_path=data\training_without_stafflines_all_classes.record 		--label_map_path=mapping.txt
python create_muscima_tf_record.py --data_dir=data\training_validation_test_without_stafflines  --set=validation 	--annotations_dir=Annotations 	--output_path=data\validation_without_stafflines_all_classes.record 	--label_map_path=mapping.txt
python create_muscima_tf_record.py --data_dir=data\training_validation_test_without_stafflines  --set=test 			--annotations_dir=Annotations 	--output_path=data\test_without_stafflines_all_classes.record 			--label_map_path=mapping.txt
                                                                                                                                                      
python create_muscima_tf_record.py --data_dir=data\training_validation_test_with_stafflines 	--set=training 		--annotations_dir=Annotations 	--output_path=data\training_with_stafflines_reduced_classes.record 		--label_map_path=mapping_reduced_class2.txt
python create_muscima_tf_record.py --data_dir=data\training_validation_test_with_stafflines 	--set=validation 	--annotations_dir=Annotations 	--output_path=data\validation_with_stafflines_reduced_classes.record 	--label_map_path=mapping_reduced_class2.txt
python create_muscima_tf_record.py --data_dir=data\training_validation_test_with_stafflines 	--set=test 			--annotations_dir=Annotations 	--output_path=data\test_with_stafflines_reduced_classes.record 			--label_map_path=mapping_reduced_class2.txt
		                                          
python create_muscima_tf_record.py --data_dir=data\training_validation_test_without_stafflines 	--set=training 		--annotations_dir=Annotations 	--output_path=data\training_without_stafflines_reduced_classes.record 	--label_map_path=mapping_reduced_class2.txt
python create_muscima_tf_record.py --data_dir=data\training_validation_test_without_stafflines 	--set=validation 	--annotations_dir=Annotations 	--output_path=data\validation_without_stafflines_reduced_classes.record --label_map_path=mapping_reduced_class2.txt
python create_muscima_tf_record.py --data_dir=data\training_validation_test_without_stafflines 	--set=test 			--annotations_dir=Annotations 	--output_path=data\test_without_stafflines_reduced_classes.record 		--label_map_path=mapping_reduced_class2.txt

Stop-Transcript
