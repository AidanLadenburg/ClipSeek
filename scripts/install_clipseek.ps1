[CmdletBinding()]
param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RequirementsFile = Join-Path $RepoRoot "requirements.txt"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
    param(
        [string]$Exe,
        [string[]]$ArgList
    )

    $shown = "$Exe $($ArgList -join ' ')"
    Write-Host $shown -ForegroundColor DarkGray
    if ($DryRun) {
        return
    }

    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $shown"
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

    foreach ($candidate in $candidates) {
        $result = Test-PythonCandidate -Exe $candidate.Exe -PrefixArgs $candidate.Args
        if ($null -ne $result) {
            return $result
        }
    }

    throw "Python was not found. Install 64-bit Python 3.10 or newer, then run this installer again."
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

    $python = Find-Python
    Write-Host "Python: $($python.ResolvedExe) ($($python.Major).$($python.Minor), $($python.Bits)-bit)"

    if ($python.Major -ne 3 -or $python.Minor -lt 10) {
        throw "ClipSeek requires Python 3.10 or newer."
    }
    if ($python.Bits -ne 64) {
        throw "ClipSeek requires 64-bit Python. Install 64-bit Python and run this again."
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

import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device: {torch.cuda.get_device_name(0)}")
"@
    Invoke-Checked -Exe $python.Exe -ArgList ($python.PrefixArgs + @("-c", $verify))

    Write-Step "Done"
    Write-Host "Installed ClipSeek libraries into this Python's user site-packages:"
    Write-Host "$($python.ResolvedExe)"
    Write-Host "No virtual environment was created and no environment variables were changed."
    Write-Host "Restart Premiere/Codex/terminals after install so they see the updated Python packages."
    exit 0
} catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
