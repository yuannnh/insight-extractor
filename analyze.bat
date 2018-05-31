@echo off
for /F "eol=# tokens=1,2 delims=	" %%a in (insights.ini) do set %%a=%%b
if not defined rootpath goto :EOF
start /B cmd.exe /c "%rscriptpath%\Rscript.exe app.R"
start http://127.0.0.1:2208