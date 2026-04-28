$logFile = "rd011_agent.log"
if (-Not (Test-Path $logFile)) { Write-Error "Log file not found."; exit }
$content = Get-Content $logFile
$assembledIndices = @()
for ($i = 0; $i -lt $content.Count; $i++) {
    if ($content[$i] -match "Document assembled:") { $assembledIndices += $i }
}
if ($assembledIndices.Count -ge 2) {
    $endIdx = $assembledIndices[-1]
    $startIdx = $assembledIndices[-2] + 1
} elseif ($assembledIndices.Count -eq 1) {
    $endIdx = $assembledIndices[-1]
    $startIdx = 0
} else {
    Write-Host "No 'Document assembled:' lines found."
    exit
}
$windowLines = $content[$startIdx..$endIdx]
$startLineNum = $startIdx + 1
$endLineNum = $endIdx + 1
$generating = @(); $generated = @(); $failed = @(); $recovered = @(); $cascade = @(); $fallback = @(); $rag = @(); $hardViolations = @(); $softWarnings = @()
for ($i = 0; $i -lt $windowLines.Count; $i++) {
    $line = $windowLines[$i]; $lnum = $startLineNum + $i
    $obj = [PSCustomObject]@{ Num = $lnum; Text = $line }
    if ($line -match "Generating section") { $generating += $obj }
    if ($line -match "Generated ") { $generated += $obj }
    if ($line -match "Generation failed for") { $failed += $obj }
    if ($line -match "Recovered |hardened same-node retry|Retrying once with hardened prompt") { $recovered += $obj }
    if ($line -match "Cascade attempt 2/") { $cascade += $obj }
    if ($line -match "Model .* failed .* trying next fallback") { $fallback += $obj }
    if ($line -match "RAG exemplars selected") { $rag += $obj }
    if ($line -match "Hard quality violations") { $hardViolations += $obj }
    if ($line -match "Soft quality warnings") { $softWarnings += $obj }
}

Write-Host "Run Metrics (Line $startLineNum to $endLineNum):"
Write-Host "- Generating section count: $($generating.Count)"
Write-Host "- Generated count: $($generated.Count)"
Write-Host "- Generation failed count: $($failed.Count)"
Write-Host "- Recovered/Hardened retry count: $($recovered.Count)"
Write-Host "- Cascade attempt 2 count: $($cascade.Count)"
Write-Host "- Model fallback count: $($fallback.Count)"
Write-Host "- RAG exemplars selected count: $($rag.Count)"
Write-Host "- Hard quality violations count: $($hardViolations.Count)"
Write-Host "- Soft quality warnings count: $($softWarnings.Count)"

Write-Host "`nEvidence Lines:"
if ($generating.Count -gt 0) {
    Write-Host "$($generating[0].Num): $($generating[0].Text)"
    if ($generating.Count -gt 1) { Write-Host "$($generating[-1].Num): $($generating[-1].Text)" }
}
foreach ($f in $failed) { Write-Host "$($f.Num): $($f.Text)" }
foreach ($r in $recovered) { Write-Host "$($r.Num): $($r.Text)" }

$matchStart = [regex]::Match($windowLines[0], "^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
$startTime = if ($matchStart.Success) { $matchStart.Value } else { "" }
$matchEnd = [regex]::Match($windowLines[-1], "^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
$endTime = if ($matchEnd.Success) { $matchEnd.Value } else { "" }
Write-Host "`nStart Timestamp: $startTime"
Write-Host "End Timestamp: $endTime"
Write-Host "Final Line ($endLineNum): $($windowLines[-1])"

$telePath = "outputs/llm_telemetry.jsonl"
if (Test-Path $telePath) {
    try {
        $telemetry = Get-Content $telePath | ForEach-Object { ConvertFrom-Json $_ }
        $windowTelemetry = $telemetry | Where-Object { $_.timestamp -ge $startTime -and $_.timestamp -le $endTime }
        if ($windowTelemetry) {
            Write-Host "`nTelemetry Summary ($($windowTelemetry.Count) entries):"
            $windowTelemetry | Group-Object task_type | Select-Object Name, Count | Format-Table -HideTableHeaders | Out-String | Write-Host
            $windowTelemetry | Group-Object provider, model | Select-Object Name, Count | Format-Table -HideTableHeaders | Out-String | Write-Host

            $errors = $windowTelemetry | Where-Object { $_.error -or ($_.status_code -and $_.status_code -ne 200) }
            Write-Host "Errors/Fallbacks: $($errors.Count)"
            if ($errors) { $errors | Group-Object error | Select-Object Name, Count | Format-Table -HideTableHeaders | Out-String | Write-Host }
        }
    } catch {
        Write-Host "Error processing telemetry"
    }
}
