$pathToGitRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF/"
$pathToSourceRoot = "C:/Users/Alex/Repositories/MusicObjectDetector-TF/MusicObjectDetector/"
$pathToTranscript = "$($pathToSourceRoot)"

Start-Transcript -path "$($pathToTranscript)PreparationTranscript.txt" -append

cd $pathToSourceRoot
echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToGitRoot)/research;$($pathToGitRoot)/research/slim;$($pathToSourceRoot)"

echo "Testing correct setup"
cd ../research
python object_detection/builders/model_builder_test.py

echo "Generating data-record in Tensorflow-format"
cd ../MusicObjectDetector

# 1. Manually copy the normalized directory into the data folder

# 2. Create the mappings
# python generate_mapping_for_normalized_dataset.py --normalized_dataset_dir ./data/normalized/muscima --mapping_output_path ./data/normalized/muscima/mapping.txt
# python generate_mapping_for_normalized_dataset.py --normalized_dataset_dir ./data/normalized/deepscores --mapping_output_path ./data/normalized/deepscores/mapping.txt
# python generate_mapping_for_normalized_dataset.py --normalized_dataset_dir ./data/normalized/mensural --mapping_output_path ./data/normalized/mensural/mapping.txt

# 3. Create the records
python create_tf_record_from_normalized_dataset.py --data_dir=data/normalized/deepscores --set=training --output_path=data/normalized/deepscores/training.record --label_map_path=data/normalized/deepscores/mapping.txt

Stop-Transcript



