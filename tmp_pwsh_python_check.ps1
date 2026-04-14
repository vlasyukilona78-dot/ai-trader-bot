[Environment]::SetEnvironmentVariable("PYTHONHOME", $null, "Process")
[Environment]::SetEnvironmentVariable("PYTHONPATH", $null, "Process")
[Environment]::SetEnvironmentVariable("VIRTUAL_ENV", $null, "Process")
Set-Location "C:\Users\Zephyrus\PycharmProjects\koteika_Ultra"
& ".\.venv\Scripts\python.exe" -V
