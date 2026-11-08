param(
    [switch]$Repair,
    [switch]$Update
)

$ErrorActionPreference = "Stop"
$Marker = "Managed by LowParLLMCapacityIncV0 installer."
$ScriptDirectory = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$LocalAppData = [Environment]::GetFolderPath("LocalApplicationData")
if ([string]::IsNullOrWhiteSpace($LocalAppData)) {
    throw "Windows did not provide a Local AppData directory for this user."
}

$ApplicationRoot = Join-Path $LocalAppData "Programs\LowP"
$EnvironmentDirectory = Join-Path $ApplicationRoot "venv"
$EnvironmentMarker = Join-Path $EnvironmentDirectory ".lowp-installed"
$ApplicationMarker = Join-Path $ApplicationRoot ".lowp-installed"
$LauncherDirectory = Join-Path $ApplicationRoot "bin"
$LauncherCommand = Join-Path $LauncherDirectory "lowp.cmd"
$LauncherScript = Join-Path $LauncherDirectory "lowp-launcher.ps1"
$PythonArguments = @()
$PythonExecutable = $null
$WindowsVersion = [Environment]::OSVersion.Version
Write-Host "Detected Windows version: $WindowsVersion"
if ($WindowsVersion.Major -lt 10) {
    throw "LowP's Windows installer requires Windows 10 or newer."
}

if ($Repair -and $Update) {
    throw "Choose either -Repair or -Update, not both."
}
if (-not (Test-Path -LiteralPath (Join-Path $ScriptDirectory "pyproject.toml") -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $ScriptDirectory "lowparllm\main.py") -PathType Leaf)) {
    throw "Run this installer from a complete LowP project checkout."
}
if (Test-Path -LiteralPath $ApplicationRoot) {
    $item = Get-Item -LiteralPath $ApplicationRoot -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to use a symbolic link or junction at $ApplicationRoot."
    }
    if (-not (Test-Path -LiteralPath $ApplicationMarker -PathType Leaf)) {
        throw "Application directory exists without a recognized LowP ownership marker: $ApplicationRoot"
    }
    $applicationMarkerItem = Get-Item -LiteralPath $ApplicationMarker -Force
    if (($applicationMarkerItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        (Get-Content -LiteralPath $ApplicationMarker -Raw).Trim() -ne $Marker) {
        throw "Application directory exists without a recognized LowP ownership marker: $ApplicationRoot"
    }
} else {
    New-Item -ItemType Directory -Path $ApplicationRoot -Force | Out-Null
    Set-Content -LiteralPath $ApplicationMarker -Value $Marker -Encoding ASCII
}

$Candidates = @()
$PythonLauncher = Get-Command "py" -ErrorAction SilentlyContinue
if ($PythonLauncher) {
    $Candidates += [pscustomobject]@{ Executable = $PythonLauncher.Source; Prefix = @("-3") }
}
$PythonCommand = Get-Command "python" -ErrorAction SilentlyContinue
if ($PythonCommand) {
    $Candidates += [pscustomobject]@{ Executable = $PythonCommand.Source; Prefix = @() }
}
foreach ($candidate in $Candidates) {
    $probeArguments = @($candidate.Prefix) + @("-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))")
    $versionText = & $candidate.Executable @probeArguments 2>$null
    if ($LASTEXITCODE -ne 0) {
        continue
    }
    try {
        $version = [version]("$versionText".Trim())
    } catch {
        continue
    }
    if ($version -ge [version]"3.11") {
        $PythonExecutable = $candidate.Executable
        $PythonArguments = @($candidate.Prefix)
        break
    }
}
if (-not $PythonExecutable) {
    throw "Python 3.11 or newer was not found. Install Python for your user, then rerun install.ps1."
}
$pipArguments = @($PythonArguments) + @("-m", "pip", "--version")
& $PythonExecutable @pipArguments
if ($LASTEXITCODE -ne 0) {
    throw "pip is unavailable for $PythonExecutable."
}

$driveRoot = [IO.Path]::GetPathRoot($LocalAppData)
$drive = [System.IO.DriveInfo]::new($driveRoot)
if ($drive.AvailableFreeSpace -lt 1GB) {
    throw "At least 1 GiB of free space is required on $driveRoot; found $([math]::Floor($drive.AvailableFreeSpace / 1GB)) GiB."
}
Write-Host ("Available disk space: {0:N1} GiB" -f ($drive.AvailableFreeSpace / 1GB))

if (Test-Path -LiteralPath $EnvironmentDirectory) {
    $environmentItem = Get-Item -LiteralPath $EnvironmentDirectory -Force
    if (($environmentItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to use a symbolic link or junction at $EnvironmentDirectory."
    }
    if (-not (Test-Path -LiteralPath $EnvironmentMarker -PathType Leaf)) {
        throw "Virtual environment exists without a recognized LowP ownership marker: $EnvironmentDirectory"
    }
    $environmentMarkerItem = Get-Item -LiteralPath $EnvironmentMarker -Force
    if (($environmentMarkerItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        (Get-Content -LiteralPath $EnvironmentMarker -Raw).Trim() -ne $Marker) {
        throw "Virtual environment exists without a recognized LowP ownership marker: $EnvironmentDirectory"
    }
} else {
    $venvArguments = @($PythonArguments) + @("-m", "venv", $EnvironmentDirectory)
    & $PythonExecutable @venvArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the LowP virtual environment."
    }
    Set-Content -LiteralPath $EnvironmentMarker -Value $Marker -Encoding ASCII
}

$EnvironmentPython = Join-Path $EnvironmentDirectory "Scripts\python.exe"
$LowpExecutable = Join-Path $EnvironmentDirectory "Scripts\lowp.exe"
if (-not (Test-Path -LiteralPath $EnvironmentPython -PathType Leaf)) {
    throw "The marked LowP environment is incomplete; refusing to replace it."
}
& $EnvironmentPython -m pip install --upgrade $ScriptDirectory
if ($LASTEXITCODE -ne 0) {
    throw "LowP dependency installation failed. Resolve the pip error and rerun this script."
}
& $EnvironmentPython -c "from lowparllm.core.config import load_config; from lowparllm.core.paths import ensure_cache_dir, ensure_data_dir; load_config(); ensure_data_dir(); ensure_cache_dir()"
if ($LASTEXITCODE -ne 0) {
    throw "Could not initialize the LowP configuration and user data directories."
}
if (-not (Test-Path -LiteralPath $LowpExecutable -PathType Leaf)) {
    throw "The installed LowP console entry point was not created."
}
& $LowpExecutable --check
if ($LASTEXITCODE -ne 0) {
    throw "The installed LowP health check failed."
}
& $LowpExecutable doctor
if ($LASTEXITCODE -ne 0) {
    throw "LowP diagnostics reported a required-component failure."
}

$trainingCheck = "import importlib.util, sys; modules = ('torch', 'transformers', 'peft', 'accelerate'); sys.exit(0 if all(importlib.util.find_spec(name) for name in modules) else 1)"
& $EnvironmentPython -c $trainingCheck 2>$null
if ($LASTEXITCODE -eq 0) {
    $trainingStatus = "installed"
} else {
    $trainingStatus = "not installed"
    $trainingChoice = Read-Host "Install optional Transformers/PEFT training dependencies? They may use several GiB [y/N]"
    if ($trainingChoice -match "^(y|yes)$") {
        $trainingTarget = $ScriptDirectory + "[training]"
        & $EnvironmentPython -m pip install --upgrade $trainingTarget
        if ($LASTEXITCODE -eq 0) {
            $trainingStatus = "installed; compatible local checkpoints and hardware are still required"
        } else {
            Write-Warning "Optional training dependencies failed to install; core LowP remains installed."
        }
    }
}
$cudaCheck = "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
& $EnvironmentPython -c $cudaCheck 2>$null
if ($LASTEXITCODE -eq 0) {
    $bitsandbytesCheck = "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('bitsandbytes') else 1)"
    & $EnvironmentPython -c $bitsandbytesCheck 2>$null
    if ($LASTEXITCODE -eq 0) {
        $qloraStatus = "bitsandbytes is installed; runtime compatibility still requires verification"
    } elseif ($trainingStatus -like "installed*") {
        $qloraChoice = Read-Host "Install optional bitsandbytes for CUDA QLoRA? Compatibility depends on your driver and PyTorch build [y/N]"
        if ($qloraChoice -match "^(y|yes)$") {
            $qloraTarget = $ScriptDirectory + "[qlora]"
            & $EnvironmentPython -m pip install --upgrade $qloraTarget
            if ($LASTEXITCODE -eq 0) {
                $qloraStatus = "installed; runtime compatibility still requires verification"
            } else {
                $qloraStatus = "installation failed; LoRA remains available if compatible"
                Write-Warning "Optional QLoRA dependency installation failed."
            }
        } else {
            $qloraStatus = "not installed; user declined optional dependencies"
        }
    } else {
        $qloraStatus = "unavailable until the optional training dependencies are installed"
    }
} else {
    $qloraStatus = "not installed; no working CUDA runtime was detected"
}

if (Test-Path -LiteralPath $LauncherDirectory) {
    $launcherDirectoryItem = Get-Item -LiteralPath $LauncherDirectory -Force
    if (($launcherDirectoryItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to use a symbolic link or junction at $LauncherDirectory."
    }
}
New-Item -ItemType Directory -Path $LauncherDirectory -Force | Out-Null
foreach ($launcher in @($LauncherCommand, $LauncherScript)) {
    if (Test-Path -LiteralPath $launcher -PathType Leaf) {
        $launcherItem = Get-Item -LiteralPath $launcher -Force
        if (($launcherItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing to overwrite a symbolic-link launcher at $launcher."
        }
        $firstLine = Get-Content -LiteralPath $launcher -TotalCount 1
        if ($firstLine -notmatch [regex]::Escape($Marker)) {
            throw "Refusing to overwrite an unrelated file at $launcher."
        }
    } elseif (Test-Path -LiteralPath $launcher) {
        throw "Refusing to overwrite a non-file launcher path at $launcher."
    }
}
$sourceLiteral = "'" + $ScriptDirectory.Replace("'", "''") + "'"
$executableLiteral = "'" + $LowpExecutable.Replace("'", "''") + "'"
$launcherScriptContent = @"
# $Marker
`$env:LOWP_SOURCE_DIR = $sourceLiteral
`$LowpExecutable = $executableLiteral
& `$LowpExecutable @args
exit `$LASTEXITCODE
"@
$launcherCommandContent = @"
rem $Marker
@echo off
where pwsh.exe >nul 2>nul
if not errorlevel 1 (
    pwsh.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0lowp-launcher.ps1" %*
) else (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0lowp-launcher.ps1" %*
)
exit /b %ERRORLEVEL%
"@
$encoding = New-Object -TypeName System.Text.UTF8Encoding -ArgumentList $false
[IO.File]::WriteAllText($LauncherScript, $launcherScriptContent, $encoding)
[IO.File]::WriteAllText($LauncherCommand, $launcherCommandContent, [Text.Encoding]::ASCII)

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$pathEntries = @($userPath -split ";" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
$launcherPath = [IO.Path]::GetFullPath($LauncherDirectory).TrimEnd("\")
$pathAlreadyPresent = $false
foreach ($entry in $pathEntries) {
    $expandedEntry = [Environment]::ExpandEnvironmentVariables($entry.Trim().Trim('"'))
    try {
        if ([string]::Equals(
            [IO.Path]::GetFullPath($expandedEntry).TrimEnd("\"),
            $launcherPath,
            [StringComparison]::OrdinalIgnoreCase
        )) {
            $pathAlreadyPresent = $true
            break
        }
    } catch {
        continue
    }
}
if (-not $pathAlreadyPresent) {
    $newUserPath = (@($pathEntries) + @($LauncherDirectory)) -join ";"
    [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
}
if ($env:Path -notlike "*$LauncherDirectory*") {
    $env:Path = "$env:Path;$LauncherDirectory"
}

Write-Host ""
Write-Host "LowP installed and verified for this Windows user."
Write-Host "Restart PowerShell or Windows Terminal if the lowp command is not yet on PATH."
Write-Host "LowP will not download, update, replace, or delete Ollama models."
Write-Host "Ollama and Git: $(if ((Get-Command ollama -ErrorAction SilentlyContinue) -and (Get-Command git -ErrorAction SilentlyContinue)) { 'available' } else { 'one or both are not installed' })"
if (Get-Command docker -ErrorAction SilentlyContinue) {
    & docker info --format "{{.ServerVersion}}" 2>$null | Out-Null
    Write-Host "Docker: $(if ($LASTEXITCODE -eq 0) { 'available' } else { 'installed; daemon unavailable' })"
} elseif (Get-Command podman -ErrorAction SilentlyContinue) {
    & podman info 2>$null | Out-Null
    Write-Host "Podman: $(if ($LASTEXITCODE -eq 0) { 'available' } else { 'installed; service unavailable' })"
} else {
    Write-Host "Docker/Podman: not installed (optional; not required for LowP)"
}
Write-Host "Optional training dependencies: $trainingStatus"
Write-Host "Optional QLoRA dependencies: $qloraStatus"
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "Installed Ollama models (read-only):"
    & ollama list
}
Write-Host "SearXNG is optional; configure a local endpoint in %APPDATA%\LowP\config.toml."
Write-Host "Application runtime: $EnvironmentDirectory"
Write-Host "User configuration and data are outside the application runtime and are preserved by uninstall."
