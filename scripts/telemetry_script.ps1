$logFile = "rd011_agent.log"
$content = Get-Content $logFile
$assembledIndices = @()
for ($i = 0; $i -lt $content.Count; $i++) {
	if ($content[$i] -match "Document assembled:") { $assembledIndices += $i }
}

$endIdx = $assembledIndices[-1]
$startIdx = if ($assembledIndices.Count -ge 2) { $assembledIndices[-2] + 1 } else { 0 }
$windowLines = $content[$startIdx..$endIdx]

$startTime = [regex]::Match($windowLines[0], "(\d{2}:\d{2}:\d{2})").Groups[1].Value
$endTime = [regex]::Match($windowLines[-1], "(\d{2}:\d{2}:\d{2})").Groups[1].Value

Write-Host "Window: $startTime - $endTime"

$telePath = "outputs/llm_telemetry.jsonl"
if (Test-Path $telePath) {
	$teleJson = Get-Content $telePath -Tail 1000 | ForEach-Object { ConvertFrom-Json $_ }
	$windowTelemetry = $teleJson | Where-Object {
		$ts = [regex]::Match($_.timestamp, "(\d{2}:\d{2}:\d{2})").Groups[1].Value
		$ts -ge $startTime -and $ts -le $endTime
	}
	if ($windowTelemetry) {
		Write-Host "`nTelemetry Summary ($($windowTelemetry.Count) entries):"
		Write-Host "By Task Type:"
		$windowTelemetry | Group-Object task_type | Select-Object Name, Count | Sort-Object Count -Descending | Format-Table -HideTableHeaders | Out-String | Write-Host
		Write-Host "By Model:"
		$windowTelemetry | Group-Object provider, model | Select-Object @{N='Model';E={"$($_.Name.Values[0])/$($_.Name.Values[1])"}}, Count | Sort-Object Count -Descending | Format-Table -HideTableHeaders | Out-String | Write-Host
		$errors = $windowTelemetry | Where-Object { $_.error -or ($_.status_code -and $_.status_code -ne 200) }
		Write-Host "Errors/Fallbacks: $($errors.Count)"
		if ($errors) { $errors | Group-Object error | Select-Object Name, Count | Format-Table -HideTableHeaders | Out-String | Write-Host }
	} else {
		Write-Host "No telemetry found."
	}
}
