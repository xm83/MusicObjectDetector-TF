Start-Transcript -path "Normalized_dimension_clustering.txt" -append

python generate_normalized_statistics.py

echo "Computing clusters for deepscores"
python dimension_clustering.py --annotations_csv_path deepscores_bounding_box_dimensions_relative.csv --visualization_width 1000 --visualization_height 800 --maximum_number_of_clusters 5

echo "Computing clusters for mensural"
python dimension_clustering.py --annotations_csv_path mensural_bounding_box_dimensions_relative.csv --visualization_width 1000 --visualization_height 800 --maximum_number_of_clusters 5

echo "Computing clusters for muscima"
python dimension_clustering.py --annotations_csv_path muscima_bounding_box_dimensions_relative.csv --visualization_width 1000 --visualization_height 800 --maximum_number_of_clusters 5

Stop-Transcript