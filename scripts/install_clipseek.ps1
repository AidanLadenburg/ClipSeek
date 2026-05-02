[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$SkipAdobeDebugMode
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RequirementsFile = Join-Path $RepoRoot "requirements.txt"
$PythonWingetId = "Python.Python.3.12"
$FallbackPythonVersion = "3.12.10"
$FallbackPythonInstallerUrl = "https://www.python.org/ftp/python/$FallbackPythonVersion/python-$FallbackPythonVersion-amd64.exe"
$FallbackPythonInstallerPath = Join-Path $env:TEMP "clipseek_python-$FallbackPythonVersion-amd64.exe"
$DefaultCsxsVersions = @("9", "10", "11", "12")

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
    param(
        [string]$Exe,
        [string[]]$ArgList,
        [int[]]$ValidExitCodes = @(0)
    )

    $shown = "$Exe $($ArgList -join ' ')"
    Write-Host $shown -ForegroundColor DarkGray
    if ($DryRun) {
        return
    }

    & $Exe @ArgList
    if ($ValidExitCodes -notcontains $LASTEXITCODE) {
        throw "Command failed with exit code ${LASTEXITCODE}: $shown"
    }
}

function Invoke-PythonScript {
    param(
        [object]$PythonInfo,
        [string]$ScriptText,
        [string]$Name
    )

    $safeName = $Name -replace '[^A-Za-z0-9_-]', '_'
    $scriptPath = Join-Path $env:TEMP "clipseek_${safeName}_$PID.py"
    try {
        Set-Content -LiteralPath $scriptPath -Value $ScriptText -Encoding UTF8
        Invoke-Checked -Exe $PythonInfo.Exe -ArgList ($PythonInfo.PrefixArgs + @($scriptPath))
    } finally {
        Remove-Item -LiteralPath $scriptPath -Force -ErrorAction SilentlyContinue
    }
}

function Test-PythonCandidate {
    param(
        [string]$Exe,
        [string[]]$PrefixArgs = @()
    )

    $probe = "import struct, sys; print(sys.executable); print(str(sys.version_info[0]) + '.' + str(sys.version_info[1])); print(struct.calcsize('P') * 8)"

    try {
        $out = & $Exe @PrefixArgs -c $probe 2>$null
        if ($LASTEXITCODE -ne 0 -or $out.Count -lt 3) {
            return $null
        }
        $parts = [string]$out[1]
        $bits = [int]$out[2]
        $ver = $parts.Split(".")
        return [pscustomobject]@{
            Exe = $Exe
            PrefixArgs = $PrefixArgs
            ResolvedExe = [string]$out[0]
            Major = [int]$ver[0]
            Minor = [int]$ver[1]
            Bits = $bits
        }
    } catch {
        return $null
    }
}

function Test-SupportedPython {
    param([object]$PythonInfo)
    if ($null -eq $PythonInfo) {
        return $false
    }
    $props = $PythonInfo.PSObject.Properties.Name
    foreach ($name in @("Major", "Minor", "Bits")) {
        if ($props -notcontains $name) {
            return $false
        }
    }
    return (
        $PythonInfo.Major -eq 3 -and
        $PythonInfo.Minor -ge 10 -and
        $PythonInfo.Bits -eq 64
    )
}

function Select-SupportedPython {
    param([object]$Candidate)
    if ($null -eq $Candidate) {
        return $null
    }
    if ($Candidate -is [array]) {
        foreach ($item in $Candidate) {
            if (Test-SupportedPython -PythonInfo $item) {
                return $item
            }
        }
        return $null
    }
    if (Test-SupportedPython -PythonInfo $Candidate) {
        return $Candidate
    }
    return $null
}

function Find-Python {
    $candidates = New-Object System.Collections.Generic.List[object]
    $candidates.Add(@{ Exe = "py"; Args = @("-3") })
    $candidates.Add(@{ Exe = "python"; Args = @() })
    $candidates.Add(@{ Exe = "python3"; Args = @() })

    foreach ($pathDir in (($env:Path -split ";") | Where-Object { $_ })) {
        $candidate = Join-Path $pathDir "python.exe"
        if (Test-Path $candidate) {
            $candidates.Add(@{ Exe = $candidate; Args = @() })
        }
    }

    $commonRoots = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python"),
        "C:\Python312",
        "C:\Python311",
        "C:\Python310"
    )
    foreach ($root in $commonRoots) {
        if (-not (Test-Path $root)) {
            continue
        }
        Get-ChildItem -Path $root -Recurse -Filter "python.exe" -ErrorAction SilentlyContinue |
            ForEach-Object {
                $candidates.Add(@{ Exe = $_.FullName; Args = @() })
            }
    }

    Get-ChildItem -Path "C:\Users" -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            $pythonRoot = Join-Path $_.FullName "AppData\Local\Programs\Python"
            if (Test-Path $pythonRoot) {
                Get-ChildItem -Path $pythonRoot -Directory -Filter "Python*" -ErrorAction SilentlyContinue |
                    ForEach-Object {
                        $exe = Join-Path $_.FullName "python.exe"
                        if (Test-Path $exe) {
                            $candidates.Add(@{ Exe = $exe; Args = @() })
                        }
                    }
            }
        }

    foreach ($registryRoot in @(
        "HKCU:\Software\Python\PythonCore",
        "HKLM:\Software\Python\PythonCore",
        "HKLM:\Software\WOW6432Node\Python\PythonCore"
    )) {
        if (-not (Test-Path $registryRoot)) {
            continue
        }
        Get-ChildItem -Path $registryRoot -ErrorAction SilentlyContinue |
            ForEach-Object {
                $installPathKey = Join-Path $_.PsPath "InstallPath"
                try {
                    $installPath = (Get-ItemProperty -Path $installPathKey -ErrorAction Stop)."(default)"
                    if ($installPath) {
                        $exe = Join-Path $installPath "python.exe"
                        if (Test-Path $exe) {
                            $candidates.Add(@{ Exe = $exe; Args = @() })
                        }
                    }
                } catch {
                    continue
                }
            }
    }

    $seen = @{}
    $bestUnsupported = $null
    foreach ($candidate in $candidates) {
        $result = Test-PythonCandidate -Exe $candidate.Exe -PrefixArgs $candidate.Args
        if ($null -ne $result) {
            $key = "$($result.ResolvedExe)|$($result.Major).$($result.Minor)|$($result.Bits)"
            if ($seen.ContainsKey($key)) {
                continue
            }
            $seen[$key] = $true

            if (Test-SupportedPython -PythonInfo $result) {
                return $result
            }
            if ($null -eq $bestUnsupported) {
                $bestUnsupported = $result
            }
        }
    }

    if ($null -ne $bestUnsupported) {
        Write-Host "Found unsupported Python: $($bestUnsupported.ResolvedExe) ($($bestUnsupported.Major).$($bestUnsupported.Minor), $($bestUnsupported.Bits)-bit)"
    }
    return $null
}

function Update-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = (@($machinePath, $userPath) | Where-Object { $_ }) -join ";"
}

function Install-Python {
    Write-Step "Installing Python"
    Write-Host "No supported Python was found. Installing 64-bit Python 3.12 for the current user."

    $winget = Get-Command "winget" -ErrorAction SilentlyContinue
    if ($null -ne $winget) {
        try {
            Write-Host "Trying winget package: $PythonWingetId"
            $null = Invoke-Checked -Exe $winget.Source -ArgList @(
                "install",
                "--id", $PythonWingetId,
                "--exact",
                "--source", "winget",
                "--scope", "user",
                "--silent",
                "--accept-package-agreements",
                "--accept-source-agreements"
            )
            Update-ProcessPath
            $python = Select-SupportedPython -Candidate (Find-Python)
            if ($null -ne $python) {
                return $python
            }
            Write-Host "winget completed, but Python was not detected. Falling back to the official installer."
        } catch {
            Write-Host "winget Python install did not complete: $($_.Exception.Message)"
            Write-Host "Falling back to the official Python installer from python.org."
        }
    } else {
        Write-Host "winget was not found. Using the official Python installer from python.org."
    }

    Write-Host "Downloading: $FallbackPythonInstallerUrl"
    Write-Host "To: $FallbackPythonInstallerPath"
    if (-not $DryRun) {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $null = Invoke-WebRequest -UseBasicParsing -Uri $FallbackPythonInstallerUrl -OutFile $FallbackPythonInstallerPath
    }

    $null = Invoke-Checked `
        -Exe $FallbackPythonInstallerPath `
        -ArgList @(
            "/quiet",
            "InstallAllUsers=0",
            "PrependPath=1",
            "Include_launcher=1",
            "Include_pip=1",
            "Include_test=0",
            "Include_doc=0",
            "Shortcuts=0"
        ) `
        -ValidExitCodes @(0, 3010)

    Update-ProcessPath
    $installed = Select-SupportedPython -Candidate (Find-Python)
    if ($null -ne $installed) {
        return $installed
    }

    if ($DryRun) {
        return [pscustomobject]@{
            Exe = "py"
            PrefixArgs = @("-3")
            ResolvedExe = "(Python 3.12 would be installed)"
            Major = 3
            Minor = 12
            Bits = 64
        }
    }

    throw "Python install finished, but a supported 64-bit Python 3.10+ could not be found. Restart Windows and run this installer again."
}

function Set-AdobeCepDebugMode {
    Write-Step "Enabling Adobe CEP panel debug mode"
    Write-Host "Setting current-user registry values so unsigned CEP panels can load."

    $adobeRoot = "HKCU:\Software\Adobe"
    $versions = New-Object System.Collections.Generic.List[string]
    foreach ($v in $DefaultCsxsVersions) {
        $versions.Add($v)
    }

    if (Test-Path $adobeRoot) {
        Get-ChildItem -Path $adobeRoot -ErrorAction SilentlyContinue |
            ForEach-Object {
                if ($_.PSChildName -match '^CSXS\.(\d+)$') {
                    if (-not $versions.Contains($Matches[1])) {
                        $versions.Add($Matches[1])
                    }
                }
            }
    }

    $orderedVersions = $versions | Sort-Object { [int]$_ } -Unique
    foreach ($v in $orderedVersions) {
        $key = Join-Path $adobeRoot "CSXS.$v"
        Write-Host "HKCU\Software\Adobe\CSXS.$v\PlayerDebugMode = 1"
        if ($DryRun) {
            continue
        }
        if (-not (Test-Path $adobeRoot)) {
            New-Item -Path $adobeRoot -Force | Out-Null
        }
        if (-not (Test-Path $key)) {
            New-Item -Path $key -Force | Out-Null
        }
        New-ItemProperty `
            -Path $key `
            -Name "PlayerDebugMode" `
            -Value "1" `
            -PropertyType String `
            -Force | Out-Null
    }
}

function Get-PytorchWheelIndex {
    $cpu = [pscustomobject]@{
        Name = "CPU"
        Index = "https://download.pytorch.org/whl/cpu"
        Reason = "No compatible NVIDIA CUDA driver was detected."
    }

    $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        return $cpu
    }

    try {
        $text = (& $nvidiaSmi.Source 2>$null | Out-String)
        if ($text -notmatch "CUDA Version:\s+(\d+)\.(\d+)") {
            return $cpu
        }

        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        $cudaValue = ($major * 10) + $minor

        if ($cudaValue -ge 128) {
            return [pscustomobject]@{
                Name = "CUDA 12.8"
                Index = "https://download.pytorch.org/whl/cu128"
                Reason = "NVIDIA driver reports CUDA $major.$minor."
            }
        }
        if ($cudaValue -ge 126) {
            return [pscustomobject]@{
                Name = "CUDA 12.6"
                Index = "https://download.pytorch.org/whl/cu126"
                Reason = "NVIDIA driver reports CUDA $major.$minor."
            }
        }
        if ($cudaValue -ge 118) {
            return [pscustomobject]@{
                Name = "CUDA 11.8"
                Index = "https://download.pytorch.org/whl/cu118"
                Reason = "NVIDIA driver reports CUDA $major.$minor."
            }
        }
    } catch {
        return $cpu
    }

    return $cpu
}

try {
    Write-Host "ClipSeek dependency installer" -ForegroundColor White
    Write-Host "Repo: $RepoRoot"

    $python = Select-SupportedPython -Candidate (Find-Python)
    if ($null -eq $python) {
        $python = Select-SupportedPython -Candidate (Install-Python)
    }
    if ($null -eq $python) {
        throw "ClipSeek requires 64-bit Python 3.10 or newer, but no supported Python was found or installed."
    }
    Write-Host "Python: $($python.ResolvedExe) ($($python.Major).$($python.Minor), $($python.Bits)-bit)"

    if (-not $SkipAdobeDebugMode) {
        Set-AdobeCepDebugMode
    } else {
        Write-Host "Skipping Adobe CEP debug-mode registry setup."
    }

    Write-Step "Upgrading pip"
    Invoke-Checked -Exe $python.Exe -ArgList ($python.PrefixArgs + @("-m", "pip", "install", "--user", "--upgrade", "pip", "setuptools", "wheel"))

    $torchIndex = Get-PytorchWheelIndex
    Write-Step "Installing PyTorch ($($torchIndex.Name))"
    Write-Host $torchIndex.Reason
    Invoke-Checked -Exe $python.Exe -ArgList ($python.PrefixArgs + @(
        "-m", "pip", "install", "--user", "--upgrade",
        "torch>=2.4,<3",
        "torchvision>=0.19,<1",
        "--index-url", $torchIndex.Index
    ))

    Write-Step "Installing NumPy 2"
    Invoke-Checked -Exe $python.Exe -ArgList ($python.PrefixArgs + @(
        "-m", "pip", "install", "--user", "--upgrade", "--force-reinstall", "--prefer-binary",
        "numpy>=2.0,<3"
    ))

    Write-Step "Installing ClipSeek Python dependencies"
    Invoke-Checked -Exe $python.Exe -ArgList ($python.PrefixArgs + @(
        "-m", "pip", "install", "--user", "--upgrade", "--prefer-binary",
        "-r", $RequirementsFile
    ))

    Write-Step "Verifying imports"
    $verify = @"
import importlib
import sys

packages = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("numpy", "numpy"),
    ("decord", "decord"),
    ("Pillow", "PIL"),
    ("einops", "einops"),
    ("safetensors", "safetensors"),
    ("faiss-cpu", "faiss"),
]

for display, module in packages:
    importlib.import_module(module)
    print(f"ok: {display}")

import numpy
import numpy._core.numeric
if int(numpy.__version__.split(".")[0]) < 2:
    raise RuntimeError(f"ClipSeek requires NumPy 2.x for embedding pickle compatibility; found {numpy.__version__}")
print(f"numpy: {numpy.__version__}")

import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device: {torch.cuda.get_device_name(0)}")
"@
    Invoke-PythonScript -PythonInfo $python -ScriptText $verify -Name "verify"

    Write-Step "Done"
    Write-Host "Installed ClipSeek libraries into this Python's user site-packages:"
    Write-Host "$($python.ResolvedExe)"
    Write-Host "No virtual environment was created."
    Write-Host "If Python was installed by this script, it was installed for the current Windows user and added to the user PATH."
    Write-Host "Restart Premiere/Codex/terminals after install so they see the updated Python packages."
    exit 0
} catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
