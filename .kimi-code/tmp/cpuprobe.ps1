param([int]$PidToProbe = 1880)
$c1 = (Get-Process -Id $PidToProbe -ErrorAction SilentlyContinue).CPU
Start-Sleep -Seconds 5
$c2 = (Get-Process -Id $PidToProbe -ErrorAction SilentlyContinue).CPU
if ($null -eq $c1 -or $null -eq $c2) { "process gone"; exit }
$delta = $c2 - $c1
"CPU delta over 5s = {0:N1}s  (healthy > 2.0s, throttled ~ 1.0s or less)" -f $delta
