$pathToSourceRoot = "C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\"
$pathToTranscript = "$($pathToSourceRoot)"

# Allowing wider outputs https://stackoverflow.com/questions/7158142/prevent-powergui-from-truncating-the-output
$pshost = get-host
$pswindow = $pshost.ui.rawui
$newsize = $pswindow.buffersize
$newsize.height = 9999
$newsize.width = 1500
$pswindow.buffersize = $newsize

cd $pathToSourceRoot
echo "Appending source root $($pathToSourceRoot) to temporary PYTHONPATH"
$env:PYTHONPATH = $pathToSourceRoot



# Compile Protoc files
Add-Type -AssemblyName System.IO.Compression.FileSystem
function Unzip
{
    param([string]$zipfile, [string]$outpath)

    [System.IO.Compression.ZipFile]::ExtractToDirectory($zipfile, $outpath)
}

$url = "https://github.com/google/protobuf/releases/download/v2.6.0/protoc-2.6.0-win32.zip"
$output = $pathToSourceRoot + "protoc-2.6.0-win32.zip"
Invoke-WebRequest -Uri $url -OutFile $output

$protoc_folder = $pathToSourceRoot + "Protoc"
Unzip $output $protoc_folder

.\Protoc\protoc.exe --version

cd ..\research

..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\anchor_generator.proto               --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\argmax_matcher.proto                 --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\bipartite_matcher.proto              --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\box_coder.proto                      --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\box_predictor.proto                  --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\eval.proto                           --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\faster_rcnn.proto                    --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\faster_rcnn_box_coder.proto          --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\grid_anchor_generator.proto          --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\hyperparams.proto                    --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\image_resizer.proto                  --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\input_reader.proto                   --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\keypoint_box_coder.proto             --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\losses.proto                         --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\matcher.proto                        --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\mean_stddev_box_coder.proto          --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\model.proto                          --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\optimizer.proto                      --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\pipeline.proto                       --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\post_processing.proto                --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\preprocessor.proto                   --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\region_similarity_calculator.proto   --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\square_box_coder.proto               --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\ssd.proto                            --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\ssd_anchor_generator.proto           --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\string_int_label_map.proto           --python_out=.
..\MusicObjectDetector\Protoc\protoc.exe object_detection\protos\train.proto                          --python_out=.