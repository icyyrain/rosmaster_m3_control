[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Car', 'Company', 'Status')]
    [string]$Mode,

    [switch]$Pause
)

$ErrorActionPreference = 'Stop'
$adapterDescription = 'Realtek Gaming 2.5GbE Family Controller'
$carPcAddress = '192.168.2.10'
$carPrefixLength = 24
$carCandidateAddresses = @('192.168.2.4', '192.168.2.3')

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-SelfElevated {
    $arguments = @(
        '-NoProfile'
        '-ExecutionPolicy', 'Bypass'
        '-File', ('"{0}"' -f $PSCommandPath)
        '-Mode', $Mode
    )
    if ($Pause) {
        $arguments += '-Pause'
    }

    $process = Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments -Wait -PassThru
    exit $process.ExitCode
}

function Get-TargetAdapter {
    $matches = @(Get-NetAdapter -IncludeHidden | Where-Object {
        $_.InterfaceDescription -eq $adapterDescription
    })

    if ($matches.Count -eq 0) {
        throw "Network adapter not found: $adapterDescription"
    }
    if ($matches.Count -gt 1) {
        throw 'Multiple matching adapters found. Stopping to avoid changing the wrong interface.'
    }
    return $matches[0]
}

function Enable-TargetAdapter {
    param([Parameter(Mandatory)]$Adapter)

    if ($Adapter.Status -eq 'Disabled') {
        Write-Host 'Enabling the Ethernet adapter...'
        $Adapter | Enable-NetAdapter -Confirm:$false
        Start-Sleep -Seconds 2
    }
}

function Remove-InterfaceIPv4Configuration {
    param([Parameter(Mandatory)][int]$InterfaceIndex)

    Get-NetRoute -InterfaceIndex $InterfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.DestinationPrefix -eq '0.0.0.0/0' } |
        Remove-NetRoute -Confirm:$false -ErrorAction SilentlyContinue

    Get-NetIPAddress -InterfaceIndex $InterfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Remove-NetIPAddress -Confirm:$false -ErrorAction SilentlyContinue
}

function Show-NetworkStatus {
    param([Parameter(Mandatory)]$Adapter)

    $adapterNow = Get-NetAdapter -InterfaceIndex $Adapter.ifIndex
    $ipInterface = Get-NetIPInterface -InterfaceIndex $Adapter.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue
    $addresses = @(Get-NetIPAddress -InterfaceIndex $Adapter.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.AddressState -ne 'Duplicate' })
    $gateway = @(Get-NetRoute -InterfaceIndex $Adapter.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.DestinationPrefix -eq '0.0.0.0/0' } |
        Sort-Object RouteMetric |
        Select-Object -First 1)
    $dns = Get-DnsClientServerAddress -InterfaceIndex $Adapter.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue

    Write-Host ''
    Write-Host 'Current network status' -ForegroundColor Cyan
    Write-Host "  Adapter: $($adapterNow.Name) / $($adapterNow.Status) / $($adapterNow.LinkSpeed)"
    if ($ipInterface) {
        Write-Host "  DHCP: $($ipInterface.Dhcp)"
    }
    if ($addresses.Count -gt 0) {
        Write-Host "  IPv4: $((($addresses | ForEach-Object { $_.IPAddress + '/' + $_.PrefixLength }) -join ', '))"
    } else {
        Write-Host '  IPv4: not assigned'
    }
    if ($gateway.Count -gt 0) {
        Write-Host "  Gateway: $($gateway[0].NextHop)"
    } else {
        Write-Host '  Gateway: none'
    }
    if ($dns -and $dns.ServerAddresses.Count -gt 0) {
        Write-Host "  DNS: $($dns.ServerAddresses -join ', ')"
    } else {
        Write-Host '  DNS: none'
    }
}

function Set-CarMode {
    param([Parameter(Mandatory)]$Adapter)

    Write-Host 'Switching to robot direct-connect mode...' -ForegroundColor Yellow
    Enable-TargetAdapter -Adapter $Adapter
    Set-NetIPInterface -InterfaceIndex $Adapter.ifIndex -AddressFamily IPv4 -Dhcp Disabled
    Remove-InterfaceIPv4Configuration -InterfaceIndex $Adapter.ifIndex
    New-NetIPAddress -InterfaceIndex $Adapter.ifIndex -AddressFamily IPv4 `
        -IPAddress $carPcAddress -PrefixLength $carPrefixLength | Out-Null
    Set-DnsClientServerAddress -InterfaceIndex $Adapter.ifIndex -ResetServerAddresses
    Clear-DnsClientCache
    Start-Sleep -Seconds 2

    Show-NetworkStatus -Adapter $Adapter
    $reachable = @($carCandidateAddresses | Where-Object {
        Test-Connection -ComputerName $_ -Count 1 -Quiet -ErrorAction SilentlyContinue
    })
    if ($reachable.Count -gt 0) {
        Write-Host "`nRobot reachable: $($reachable -join ', ')" -ForegroundColor Green
    } else {
        Write-Host "`nConfiguration complete, but the robot did not reply. Check the cable and robot power; expected address: 192.168.2.3 or 192.168.2.4." -ForegroundColor Yellow
    }
}

function Set-CompanyMode {
    param([Parameter(Mandatory)]$Adapter)

    Write-Host 'Switching to company DHCP mode...' -ForegroundColor Yellow
    Enable-TargetAdapter -Adapter $Adapter
    Remove-InterfaceIPv4Configuration -InterfaceIndex $Adapter.ifIndex
    Set-NetIPInterface -InterfaceIndex $Adapter.ifIndex -AddressFamily IPv4 -Dhcp Enabled
    Set-DnsClientServerAddress -InterfaceIndex $Adapter.ifIndex -ResetServerAddresses
    Get-NetAdapter -InterfaceIndex $Adapter.ifIndex | Restart-NetAdapter -Confirm:$false
    Start-Sleep -Seconds 3

    & ipconfig.exe /renew $Adapter.Name | Out-Null
    for ($attempt = 0; $attempt -lt 15; $attempt++) {
        $dhcpAddress = Get-NetIPAddress -InterfaceIndex $Adapter.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.PrefixOrigin -eq 'Dhcp' -and $_.AddressState -eq 'Preferred' } |
            Select-Object -First 1
        if ($dhcpAddress) {
            break
        }
        Start-Sleep -Seconds 1
    }
    Clear-DnsClientCache

    Show-NetworkStatus -Adapter $Adapter
    if ($dhcpAddress) {
        Write-Host "`nCompany network address acquired: $($dhcpAddress.IPAddress)" -ForegroundColor Green
    } else {
        Write-Host "`nNo DHCP address acquired. Check the company cable, switch port, or network authentication." -ForegroundColor Yellow
    }
}

try {
    if (-not (Test-IsAdministrator)) {
        Invoke-SelfElevated
    }

    $adapter = Get-TargetAdapter
    switch ($Mode) {
        'Car'     { Set-CarMode -Adapter $adapter }
        'Company' { Set-CompanyMode -Adapter $adapter }
        'Status'  { Show-NetworkStatus -Adapter $adapter }
    }
    $exitCode = 0
} catch {
    Write-Host "`nSwitch failed: $($_.Exception.Message)" -ForegroundColor Red
    $exitCode = 1
}

if ($Pause) {
    Write-Host ''
    Read-Host 'Press Enter to close'
}
exit $exitCode
