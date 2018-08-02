$pathToSourceRoot = "C:\Users\Alex\Repositories\MusicObjectDetector-TF\MusicObjectDetector\"
$pathToTranscript = "$($pathToSourceRoot)"

cd $pathToSourceRoot
echo "Appending source root $($pathToSourceRoot) to temporary PYTHONPATH"
$env:PYTHONPATH = $pathToSourceRoot

Start-Transcript -path "$($pathToTranscript)Transcript.txt" -append

# Compile Protoc files
Add-Type -AssemblyName System.IO.Compression.FileSystem
function Unzip
{
    param([string]$zipfile, [string]$outpath)

    [System.IO.Compression.ZipFile]::ExtractToDirectory($zipfile, $outpath)
}

$url = "https://github.com/google/protobuf/releases/download/v3.4.0/protoc-3.4.0-win32.zip"
$output = $pathToSourceRoot + "protoc-3.4.0-win32.zip"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -Uri $url -OutFile $output

$protoc_folder = $pathToSourceRoot + "Protoc"
Unzip $output $protoc_folder

.\Protoc\bin\protoc.exe --version

cd ..\research
..\MusicObjectDetector\Protoc\bin\protoc.exe object_detection/protos/*.proto --python_out=.

rm ..\MusicObjectDetector\Protoc -Recurse
rm ..\MusicObjectDetector\protoc-3.4.0-win32.zip

echo "Verifying correct installation..."
python object_detection\builders\model_builder_test.py

Stop-Transcript