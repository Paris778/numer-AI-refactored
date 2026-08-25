$os = Get-CimInstance Win32_OperatingSystem
"TotalGB={0:N1} FreeGB={1:N1}" -f ($os.TotalVisibleMemorySize/1MB), ($os.FreePhysicalMemory/1MB)
"--- python processes ---"
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
if ($procs) {
  $procs | Select-Object ProcessId, @{n='WS_GB';e={[math]::Round($_.WorkingSetSize/1GB,1)}} | Format-Table -AutoSize | Out-String
} else {
  "none"
}
"--- CPU cores ---"
(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
