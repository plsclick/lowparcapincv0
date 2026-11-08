param()

$ErrorActionPreference = "Stop"
$Marker = "Managed by LowParLLMCapacityIncV0 installer."
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

if (Test-Path -LiteralPath $ApplicationRoot) {
    $applicationItem = Get-Item -LiteralPath $ApplicationRoot -Force
    if (($applicationItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to inspect a symbolic link or junction at $ApplicationRoot."
    }
}
$runtimeOwned = $false
$applicationMarkerOwned = $false
if (Test-Path -LiteralPath $ApplicationMarker -PathType Leaf) {
    $markerItem = Get-Item -LiteralPath $ApplicationMarker -Force
    if (($markerItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to read a symbolic-link ownership marker at $ApplicationMarker."
    }
    $applicationMarkerOwned = (Get-Content -LiteralPath $ApplicationMarker -Raw).Trim() -eq $Marker
    $runtimeOwned = $applicationMarkerOwned
}
if (Test-Path -LiteralPath $EnvironmentDirectory) {
    $item = Get-Item -LiteralPath $EnvironmentDirectory -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to remove a symbolic link or junction at $EnvironmentDirectory."
    }
    if (-not (Test-Path -LiteralPath $EnvironmentMarker -PathType Leaf)) {
        throw "Virtual environment ownership could not be verified; nothing was removed."
    }
    $environmentMarkerItem = Get-Item -LiteralPath $EnvironmentMarker -Force
    if (($environmentMarkerItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        (Get-Content -LiteralPath $EnvironmentMarker -Raw).Trim() -ne $Marker) {
        throw "Virtual environment ownership could not be verified; nothing was removed."
    }
    $runtimeOwned = $true
}

$ownedLaunchers = @()
if (Test-Path -LiteralPath $LauncherDirectory) {
    $launcherDirectoryItem = Get-Item -LiteralPath $LauncherDirectory -Force
    if (($launcherDirectoryItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to inspect a symbolic link or junction at $LauncherDirectory."
    }
}
foreach ($launcher in @($LauncherCommand, $LauncherScript)) {
    if (Test-Path -LiteralPath $launcher -PathType Leaf) {
        $launcherItem = Get-Item -LiteralPath $launcher -Force
        if (($launcherItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing to remove a symbolic-link launcher at $launcher."
        }
        $firstLine = Get-Content -LiteralPath $launcher -TotalCount 1
        if ($firstLine -match [regex]::Escape($Marker)) {
            $ownedLaunchers += $launcher
        }
    }
}
if (-not $runtimeOwned -and $ownedLaunchers.Count -eq 0) {
    Write-Host "No LowP-owned Windows runtime or launcher was found. Nothing was removed."
    exit 0
}

Write-Host "The following LowP-owned application files will be removed:"
if ($runtimeOwned) {
    Write-Host "  Application virtual environment: $EnvironmentDirectory"
}
foreach ($launcher in $ownedLaunchers) {
    Write-Host "  Launcher: $launcher"
}
Write-Host "Configuration, models, adapters, training data, knowledge databases, sessions, and projects will be preserved."
$runningProcesses = @()
if ($runtimeOwned -and (Test-Path -LiteralPath $EnvironmentDirectory -PathType Container)) {
    $runtimeScripts = [IO.Path]::GetFullPath((Join-Path $EnvironmentDirectory "Scripts")).TrimEnd("\") + "\"
    try {
        $runningProcesses = @(
            Get-CimInstance -ClassName Win32_Process | Where-Object {
                $_.ExecutablePath -and
                [IO.Path]::GetFullPath($_.ExecutablePath).StartsWith(
                    $runtimeScripts,
                    [StringComparison]::OrdinalIgnoreCase
                )
            }
        )
    } catch {
        throw "Could not safely inspect LowP-owned processes; no files were removed. $($_.Exception.Message)"
    }
}
if ($runningProcesses.Count -gt 0) {
    Write-Host "  Active LowP virtual-environment processes to stop after confirmation: $($runningProcesses.Count)"
}
$confirmation = Read-Host "Type REMOVE to confirm"
if ($confirmation -cne "REMOVE") {
    Write-Host "Uninstallation cancelled."
    exit 0
}

foreach ($process in $runningProcesses) {
    try {
        Stop-Process -Id ([int]$process.ProcessId) -Force -ErrorAction Stop
    } catch {
        throw "Could not stop verified LowP process $($process.ProcessId); no application files were removed. $($_.Exception.Message)"
    }
}
foreach ($launcher in $ownedLaunchers) {
    Remove-Item -LiteralPath $launcher -Force
}
if ($runtimeOwned -and (Test-Path -LiteralPath $EnvironmentDirectory)) {
    Remove-Item -LiteralPath $EnvironmentDirectory -Recurse -Force
}
if ($applicationMarkerOwned) {
    Remove-Item -LiteralPath $ApplicationMarker -Force
}
if (Test-Path -LiteralPath $LauncherDirectory -PathType Container) {
    if (@(Get-ChildItem -LiteralPath $LauncherDirectory -Force).Count -eq 0) {
        Remove-Item -LiteralPath $LauncherDirectory -Force
    }
}
if (Test-Path -LiteralPath $ApplicationRoot -PathType Container) {
    if (@(Get-ChildItem -LiteralPath $ApplicationRoot -Force).Count -eq 0) {
        Remove-Item -LiteralPath $ApplicationRoot -Force
    }
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$originalUserPath = $userPath
$keptEntries = @()
foreach ($entry in @($userPath -split ";" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
    $expandedEntry = [Environment]::ExpandEnvironmentVariables($entry.Trim().Trim('"'))
    try {
        $isLauncherPath = [string]::Equals(
            [IO.Path]::GetFullPath($expandedEntry).TrimEnd("\"),
            [IO.Path]::GetFullPath($LauncherDirectory).TrimEnd("\"),
            [StringComparison]::OrdinalIgnoreCase
        )
    } catch {
        $isLauncherPath = $false
    }
    if (-not $isLauncherPath) {
        $keptEntries += $entry
    }
}
if (-not [string]::IsNullOrWhiteSpace($originalUserPath) -and
    ($keptEntries -join ";") -ne $originalUserPath) {
    [Environment]::SetEnvironmentVariable("Path", ($keptEntries -join ";"), "User")
}
Write-Host "LowP runtime and launchers removed. User data was preserved."
Write-Host "Restart PowerShell or Windows Terminal to refresh PATH."
