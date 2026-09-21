@echo off
set "STM32_CLI=%LOCALAPPDATA%\STMicroelectronics\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe"

if not exist "%STM32_CLI%" (
  echo STM32CubeProgrammer CLI was not found:
  echo   %STM32_CLI%
  exit /b 1
)

"%STM32_CLI%" %*
