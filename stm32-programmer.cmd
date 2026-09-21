@echo off
rem Locate STM32CubeProgrammer CLI. The normal per-user install location is
rem checked first; the second path is where an installer run inside the OpenAI
rem Codex MSIX container redirects its writes.
set "STM32_CLI=%LOCALAPPDATA%\STMicroelectronics\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe"

if not exist "%STM32_CLI%" (
  set "STM32_CLI=%LOCALAPPDATA%\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Local\STMicroelectronics\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe"
)

if not exist "%STM32_CLI%" (
  set "STM32_CLI=%ProgramFiles%\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe"
)

if not exist "%STM32_CLI%" (
  echo STM32CubeProgrammer CLI was not found in any known location.
  echo Reinstall from vendor\yahboom_m3pro_official\tools\en.stm32cubeprg-win64-v2-19-0.zip
  exit /b 1
)

"%STM32_CLI%" %*
