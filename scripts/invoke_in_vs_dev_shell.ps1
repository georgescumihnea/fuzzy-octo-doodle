param(
    [Parameter(Mandatory = $true)]
    [string]$Command
)

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe was not found."
}

$installationPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installationPath) {
    throw "Visual Studio with C++ build tools was not found."
}

$vsDevCmd = Join-Path $installationPath "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $vsDevCmd)) {
    throw "VsDevCmd.bat was not found at $vsDevCmd."
}

$vulkanSdkRoot = "C:\VulkanSDK"
if (-not $env:VULKAN_SDK -and (Test-Path $vulkanSdkRoot)) {
    $latestSdk = Get-ChildItem $vulkanSdkRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if ($latestSdk) {
        $env:VULKAN_SDK = $latestSdk.FullName
        $sdkBin = Join-Path $env:VULKAN_SDK "Bin"
        if (Test-Path $sdkBin) {
            $env:Path = "$sdkBin;$env:Path"
        }
    }
}

$cmdLine = "`"$vsDevCmd`" -arch=x64 -host_arch=x64 && $Command"
cmd.exe /d /s /c $cmdLine

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
