setlocal EnableDelayedExpansion

mkdir %TMP_DIR_WIN%\bin

if "%REBUILD%"=="" (
  IF EXIST %TMP_DIR_WIN%\bin\sccache.exe (
    taskkill /im sccache.exe /f /t || ver > nul
    del %TMP_DIR_WIN%\bin\sccache.exe || ver > nul
  )
  set USE_CURL=0
  if "%BUILD_ENVIRONMENT%"=="" (
    set USE_CURL=1
  ) else (
    aws sts get-caller-identity >nul 2>&1
    if errorlevel 1 (
      echo No valid AWS credentials, fallback to curl request
      set USE_CURL=1
    )
  )
  if "!USE_CURL!"=="1" (
    curl --retry 3 --retry-all-errors -k https://s3.amazonaws.com/ossci-windows/sccache-v0.7.4.exe --output %TMP_DIR_WIN%\bin\sccache.exe
  ) else (
    aws s3 cp s3://ossci-windows/sccache-v0.7.4.exe %TMP_DIR_WIN%\bin\sccache.exe
  )
)
