param([Parameter(Mandatory=$true)][string]$Manifest,[Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null=[Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
$null=[Windows.Graphics.Imaging.BitmapDecoder,Windows.Foundation,ContentType=WindowsRuntime]
$null=[Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
$asTask=([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {$_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'})[0]
function Await($operation,[Type]$type) {
    $task=$asTask.MakeGenericMethod($type).Invoke($null,@($operation)); $task.Wait(); return $task.Result
}
$engine=[Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new('en-US'))
if ($null -eq $engine) {throw 'Installed English OCR unavailable'}
$rows=Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
$results=@()
foreach ($row in $rows) {
    try {
        $file=Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($row.path)) ([Windows.Storage.StorageFile])
        $stream=Await ($file.OpenReadAsync()) ([Windows.Storage.Streams.IRandomAccessStreamWithContentType])
        $decoder=Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $bitmap=Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
        $result=Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
        $words=@(foreach ($line in $result.Lines) {foreach ($word in $line.Words) {@{text=$word.Text;box=@($word.BoundingRect.X,$word.BoundingRect.Y,$word.BoundingRect.Width,$word.BoundingRect.Height)}}})
        $results+=@{id=$row.id;text=$result.Text;words=$words;failure=$null}
        $bitmap.Dispose(); $stream.Dispose()
    } catch {$results+=@{id=$row.id;text='';words=@();failure=$_.Exception.Message}}
}
ConvertTo-Json -InputObject $results -Depth 8 | Set-Content -LiteralPath $Output -Encoding utf8
