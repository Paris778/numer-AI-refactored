$p = Get-Process -Id 1880 -ErrorAction SilentlyContinue
if ($null -eq $p) { "process gone"; exit }
"total CPU so far: {0:N0}s = {1:N1}h" -f $p.CPU, ($p.CPU/3600)
"elapsed: {0:N1}h" -f ((Get-Date) - $p.StartTime).TotalHours
