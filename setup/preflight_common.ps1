function Convert-ToVersionOrNull {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }

    try {
        return [version]$Value
    } catch {
        return $null
    }
}

function Test-AnyPatternMatch {
    param(
        [string[]]$Texts,
        [string[]]$Patterns
    )

    foreach ($text in $Texts) {
        if ([string]::IsNullOrWhiteSpace($text)) {
            continue
        }

        foreach ($pattern in $Patterns) {
            if ($text -match $pattern) {
                return $true
            }
        }
    }

    return $false
}

function Get-PlatformProfile {
    param(
        [string]$CpuName,
        [object[]]$Profiles
    )

    foreach ($profile in $Profiles) {
        if ($CpuName -match [string]$profile.match) {
            return $profile
        }
    }

    return $null
}

function Get-FormattedBiosDate {
    param([string]$RawValue)

    if ([string]::IsNullOrWhiteSpace($RawValue)) {
        return 'unknown'
    }

    try {
        return ([Management.ManagementDateTimeConverter]::ToDateTime($RawValue)).ToString('yyyy-MM-dd')
    } catch {
        return $RawValue
    }
}

function Get-CompatibilityConfig {
    param([string]$CompatibilityFile)

    if ([string]::IsNullOrWhiteSpace($CompatibilityFile)) {
        $CompatibilityFile = Join-Path $PSScriptRoot 'compatibility.json'
    }

    if (-not (Test-Path $CompatibilityFile)) {
        throw "Compatibility file not found: $CompatibilityFile"
    }

    return Get-Content -Path $CompatibilityFile -Raw | ConvertFrom-Json
}

function Get-NpuDriverInfo {
    param(
        [string[]]$NamePatterns,
        [string[]]$SearchTerms
    )

    try {
        $pnpDeviceCommand = Get-Command Get-PnpDevice -ErrorAction SilentlyContinue
        if ($pnpDeviceCommand) {
            $pnpDevice = Get-PnpDevice -PresentOnly -ErrorAction Stop | Where-Object {
                $texts = @($_.FriendlyName, $_.Class, $_.InstanceId)
                Test-AnyPatternMatch -Texts $texts -Patterns $NamePatterns
            } | Select-Object -First 1

            if ($pnpDevice) {
                $driverVersion = $null
                $manufacturer = $null
                $driverPropertyCommand = Get-Command Get-PnpDeviceProperty -ErrorAction SilentlyContinue

                if ($driverPropertyCommand) {
                    try {
                        $driverVersion = [string](Get-PnpDeviceProperty -InstanceId $pnpDevice.InstanceId -KeyName 'DEVPKEY_Device_DriverVersion' -ErrorAction Stop).Data
                    } catch {
                    }

                    try {
                        $manufacturer = [string](Get-PnpDeviceProperty -InstanceId $pnpDevice.InstanceId -KeyName 'DEVPKEY_Device_DriverProvider' -ErrorAction Stop).Data
                    } catch {
                    }
                }

                return [pscustomobject]@{
                    DeviceName = [string]$pnpDevice.FriendlyName
                    DriverVersion = $driverVersion
                    Manufacturer = $manufacturer
                }
            }
        }
    } catch {
    }

    foreach ($searchTerm in $SearchTerms) {
        $escapedSearchTerm = $searchTerm.Replace("'", "''")
        $filter = "DeviceName LIKE '%$escapedSearchTerm%'"

        try {
            $driver = Get-CimInstance Win32_PnPSignedDriver -Filter $filter -ErrorAction Stop | Select-Object -First 1
            if ($driver) {
                return [pscustomobject]@{
                    DeviceName = [string]$driver.DeviceName
                    DriverVersion = [string]$driver.DriverVersion
                    Manufacturer = [string]$driver.Manufacturer
                }
            }
        } catch {
        }
    }

    return $null
}

function New-HardwareSnapshot {
    param(
        [string]$CpuName,
        [string]$WindowsProduct,
        [int]$WindowsBuild,
        [string]$BiosVendor,
        [string]$BiosVersion,
        [string]$BiosDate,
        [object]$Driver
    )

    return [pscustomobject]@{
        CpuName = $CpuName
        WindowsProduct = $WindowsProduct
        WindowsBuild = $WindowsBuild
        BiosVendor = $BiosVendor
        BiosVersion = $BiosVersion
        BiosDate = $BiosDate
        Driver = $Driver
    }
}

function Get-LocalHardwareSnapshot {
    param(
        [pscustomobject]$Compatibility,
        [switch]$SkipDriverCheck
    )

    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $bios = Get-CimInstance Win32_BIOS | Select-Object -First 1
    $osInfo = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'

    $cpuName = ([string]$cpu.Name).Trim()
    $biosVendor = [string]$bios.Manufacturer
    $biosVersion = [string]$bios.SMBIOSBIOSVersion
    $biosDate = Get-FormattedBiosDate -RawValue ([string]$bios.ReleaseDate)
    $windowsProduct = [string]$osInfo.ProductName
    $windowsBuild = 0
    [void][int]::TryParse([string]$osInfo.CurrentBuildNumber, [ref]$windowsBuild)

    if ($windowsBuild -ge 22000 -and $windowsProduct -match '^Windows 10') {
        $windowsProduct = $windowsProduct -replace '^Windows 10', 'Windows 11'
    }

    $driver = $null
    if (-not $SkipDriverCheck) {
        $namePatterns = @()
        if ($Compatibility.PSObject.Properties.Name -contains 'driverNamePatterns') {
            $namePatterns = @($Compatibility.driverNamePatterns)
        }

        $searchTerms = @()
        if ($Compatibility.PSObject.Properties.Name -contains 'driverSearchTerms') {
            $searchTerms = @($Compatibility.driverSearchTerms)
        }

        $driver = Get-NpuDriverInfo -NamePatterns $namePatterns -SearchTerms $searchTerms
    }

    return New-HardwareSnapshot -CpuName $cpuName -WindowsProduct $windowsProduct -WindowsBuild $windowsBuild -BiosVendor $biosVendor -BiosVersion $biosVersion -BiosDate $biosDate -Driver $driver
}

function Test-CompatibilityComboMatch {
    param(
        [pscustomobject]$Snapshot,
        [pscustomobject]$Combo
    )

    if (-not $Combo) {
        return $false
    }

    if ($Combo.PSObject.Properties.Name -contains 'cpuRegex') {
        $cpuRegex = [string]$Combo.cpuRegex
        if (-not [string]::IsNullOrWhiteSpace($cpuRegex) -and $Snapshot.CpuName -notmatch $cpuRegex) {
            return $false
        }
    }

    if ($Combo.PSObject.Properties.Name -contains 'biosVendorRegex') {
        $biosVendorRegex = [string]$Combo.biosVendorRegex
        if (-not [string]::IsNullOrWhiteSpace($biosVendorRegex) -and $Snapshot.BiosVendor -notmatch $biosVendorRegex) {
            return $false
        }
    }

    if ($Combo.PSObject.Properties.Name -contains 'biosVersionRegex') {
        $biosVersionRegex = [string]$Combo.biosVersionRegex
        if (-not [string]::IsNullOrWhiteSpace($biosVersionRegex) -and $Snapshot.BiosVersion -notmatch $biosVersionRegex) {
            return $false
        }
    }

    if ($Combo.PSObject.Properties.Name -contains 'driverVersionRegex') {
        $driverVersionRegex = [string]$Combo.driverVersionRegex
        if (-not [string]::IsNullOrWhiteSpace($driverVersionRegex)) {
            if (-not $Snapshot.Driver -or [string]$Snapshot.Driver.DriverVersion -notmatch $driverVersionRegex) {
                return $false
            }
        }
    }

    if ($Combo.PSObject.Properties.Name -contains 'driverVersion') {
        $driverVersion = [string]$Combo.driverVersion
        if (-not [string]::IsNullOrWhiteSpace($driverVersion)) {
            if (-not $Snapshot.Driver -or [string]$Snapshot.Driver.DriverVersion -ne $driverVersion) {
                return $false
            }
        }
    }

    return $true
}

function Get-CompatibilityMatches {
    param(
        [pscustomobject]$Snapshot,
        [pscustomobject]$Compatibility
    )

    $validated = @()
    $validatedCombos = @()
    if ($Compatibility.PSObject.Properties.Name -contains 'knownValidatedCombos') {
        $validatedCombos = @($Compatibility.knownValidatedCombos)
    }
    foreach ($combo in $validatedCombos) {
        if (Test-CompatibilityComboMatch -Snapshot $Snapshot -Combo $combo) {
            $validated += $combo
        }
    }

    $problem = @()
    $problemCombos = @()
    if ($Compatibility.PSObject.Properties.Name -contains 'knownProblemCombos') {
        $problemCombos = @($Compatibility.knownProblemCombos)
    }
    foreach ($combo in $problemCombos) {
        if (Test-CompatibilityComboMatch -Snapshot $Snapshot -Combo $combo) {
            $problem += $combo
        }
    }

    return [pscustomobject]@{
        Validated = $validated
        Problem = $problem
    }
}

function Test-HardwareCompatibility {
    param(
        [pscustomobject]$Snapshot,
        [pscustomobject]$Compatibility,
        [switch]$SkipDriverCheck
    )

    $profile = Get-PlatformProfile -CpuName $Snapshot.CpuName -Profiles @($Compatibility.cpuProfiles)
    $errors = New-Object System.Collections.Generic.List[string]
    $warnings = New-Object System.Collections.Generic.List[string]
    $notes = New-Object System.Collections.Generic.List[string]
    $requiredEnv = @{}

    $minimumWindowsBuild = [int]$Compatibility.minimumWindowsBuild
    if ($Snapshot.WindowsBuild -lt $minimumWindowsBuild) {
        $errors.Add("Windows build $($Snapshot.WindowsBuild) detected. Windows 11 build $minimumWindowsBuild or newer is required.")
    }

    if (-not $profile) {
        $errors.Add("Unsupported CPU: '$($Snapshot.CpuName)'. This repo currently targets Intel Core Ultra systems with an Intel NPU.")
    } elseif ([string]$profile.id -eq 'core-ultra-generic') {
        $warnings.Add('Core Ultra CPU detected but not mapped to a known profile yet. Continuing without platform-specific overrides.')
    }

    if (-not $SkipDriverCheck) {
        if (-not $Snapshot.Driver) {
            $errors.Add('Intel NPU device/driver not detected. Install the Intel NPU driver and verify the device appears in Device Manager.')
        } else {
            $minimumDriverVersion = Convert-ToVersionOrNull ([string]$Compatibility.minimumNpuDriverVersion)
            $detectedDriverVersion = Convert-ToVersionOrNull ([string]$Snapshot.Driver.DriverVersion)

            if (-not $detectedDriverVersion) {
                $warnings.Add('Intel NPU device detected, but the driver version could not be read automatically.')
            } elseif ($minimumDriverVersion -and $detectedDriverVersion -lt $minimumDriverVersion) {
                $errors.Add("Intel NPU driver $detectedDriverVersion detected. Version $minimumDriverVersion or newer is required.")
            }
        }
    }

    foreach ($rule in @($Compatibility.blockedBiosVersions)) {
        $vendorRegex = [string]$rule.vendorRegex
        $versionRegex = [string]$rule.versionRegex
        $vendorMatches = $true
        $versionMatches = $true

        if (-not [string]::IsNullOrWhiteSpace($vendorRegex)) {
            $vendorMatches = $Snapshot.BiosVendor -match $vendorRegex
        }

        if (-not [string]::IsNullOrWhiteSpace($versionRegex)) {
            $versionMatches = $Snapshot.BiosVersion -match $versionRegex
        }

        if ($vendorMatches -and $versionMatches) {
            $message = [string]$rule.message
            if ([string]::IsNullOrWhiteSpace($message)) {
                $message = "BIOS version '$($Snapshot.BiosVersion)' is blocked by setup\\compatibility.json."
            }
            $errors.Add($message)
        }
    }

    $matches = Get-CompatibilityMatches -Snapshot $Snapshot -Compatibility $Compatibility
    foreach ($combo in @($matches.Validated)) {
        $label = [string]$combo.label
        if ([string]::IsNullOrWhiteSpace($label)) {
            $label = 'Matched a known validated hardware/driver combination.'
        }
        $notes.Add($label)
    }

    foreach ($combo in @($matches.Problem)) {
        $message = [string]$combo.message
        if ([string]::IsNullOrWhiteSpace($message)) {
            $message = 'Matched a known problematic hardware/driver combination.'
        }

        $severity = ([string]$combo.severity).ToLowerInvariant()
        if ($severity -eq 'warning') {
            $warnings.Add($message)
        } else {
            $errors.Add($message)
        }
    }

    if ($profile -and $profile.requiredEnv) {
        foreach ($property in $profile.requiredEnv.PSObject.Properties) {
            $requiredEnv[$property.Name] = [string]$property.Value
        }
    }

    return [pscustomobject]@{
        Passed = ($errors.Count -eq 0)
        CpuName = $Snapshot.CpuName
        WindowsProduct = $Snapshot.WindowsProduct
        WindowsBuild = $Snapshot.WindowsBuild
        BiosVendor = $Snapshot.BiosVendor
        BiosVersion = $Snapshot.BiosVersion
        BiosDate = $Snapshot.BiosDate
        Driver = $Snapshot.Driver
        Profile = $profile
        RequiredEnv = $requiredEnv
        Errors = @($errors)
        Warnings = @($warnings)
        Notes = @($notes)
        Snapshot = $Snapshot
        MatchedValidatedCombos = @($matches.Validated)
        MatchedProblemCombos = @($matches.Problem)
        MinimumWindowsBuild = [int]$Compatibility.minimumWindowsBuild
        MinimumNpuDriverVersion = [string]$Compatibility.minimumNpuDriverVersion
    }
}

function Apply-CompatibilityEnvironment {
    param([pscustomobject]$Report)

    foreach ($key in @($Report.RequiredEnv.Keys)) {
        Set-Item -Path ("Env:{0}" -f $key) -Value ([string]$Report.RequiredEnv[$key])
    }
}