# Verifies the system Tesseract install can see Urdu traineddata.
# Does not install anything - just reports, since language data placement
# varies by how Tesseract was installed.

$tesseract = Get-Command tesseract -ErrorAction SilentlyContinue
if (-not $tesseract) {
    Write-Output "WARNING: tesseract executable not found on PATH. Install Tesseract OCR (with Urdu language data) from https://github.com/UB-Mannheim/tesseract/wiki"
    exit 0
}

$langs = & tesseract --list-langs 2>&1
if ($langs -match "urd") {
    Write-Output "OK: Tesseract Urdu (urd) language data found."
} else {
    Write-Output "WARNING: Tesseract is installed but 'urd' language data was not found."
    Write-Output "Download urd.traineddata from https://github.com/tesseract-ocr/tessdata and place it in the tessdata folder shown by: tesseract --print-parameters | findstr tessdata"
}
