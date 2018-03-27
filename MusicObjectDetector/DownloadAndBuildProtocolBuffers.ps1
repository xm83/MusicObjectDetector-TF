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

Start-Transcript -path "$($pathToTranscript)Transcript.txt" -append

# Compile Protoc files
Add-Type -AssemblyName System.IO.Compression.FileSystem
function Unzip
{
    param([string]$zipfile, [string]$outpath)

    [System.IO.Compression.ZipFile]::ExtractToDirectory($zipfile, $outpath)
}

# $url = "https://github.com/google/protobuf/releases/download/v2.6.0/protoc-2.6.0-win32.zip"
# $output = $pathToSourceRoot + "protoc-2.6.0-win32.zip"

$url = "https://github.com/google/protobuf/releases/download/v3.4.0/protoc-3.4.0-win32.zip"
$output = $pathToSourceRoot + "protoc-3.4.0-win32.zip"
Invoke-WebRequest -Uri $url -OutFile $output

$protoc_folder = $pathToSourceRoot + "Protoc"
Unzip $output $protoc_folder

.\Protoc\bin\protoc.exe --version

cd ..\research

..\MusicObjectDetector\Protoc\bin\protoc.exe object_detection\protos\*.proto               --python_out=.

Stop-Transcript