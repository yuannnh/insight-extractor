<# : chooser.bat
:: launches a File... Open sort of file chooser and outputs choice(s) to the console
:: https://stackoverflow.com/a/15885133/1683264

@echo off
setlocal

for /F "eol=# tokens=1,2 delims=	" %%a in (insights.ini) do set %%a=%%b

if not defined datapath goto :EOF
cd %datapath%

for /f "delims=" %%I in ('powershell -noprofile "iex (${%~f0} | out-string)"') do (
    set "datafile=%%~I"
	set "filepath=%%~pI"
)

if not defined datafile goto :EOF
if not defined rootpath goto :EOF
cd %rootpath%
set t=%filepath%

:loop
for /f "tokens=1* delims=\" %%a in ("%t%") do (
   set connector=%%a
   set t=%%b
)
if defined t goto :loop

echo python syntagm_extractor.py %connector% %datafile%
python syntagm_extractor.py "%connector%" "%datafile%"
start /B cmd.exe /c "%rscriptpath%\Rscript.exe app.R"
timeout 3 > nul
start http://127.0.0.1:2208
goto :EOF

: end Batch portion / begin PowerShell hybrid chimera #>

Add-Type -AssemblyName System.Windows.Forms
$f = new-object Windows.Forms.OpenFileDialog
$f.InitialDirectory = pwd
$f.Filter = "CSV Files (*.csv)|*.csv|Excel Files (*.xls)|*.xls|Web Crawl (*.web)|*.web|Text Files (*.txt)|*.txt|All Files (*.*)|*.*"
$f.ShowHelp = $true
$f.Multiselect = $true
[void]$f.ShowDialog()
if ($f.Multiselect) { $f.FileNames } else { $f.FileName }