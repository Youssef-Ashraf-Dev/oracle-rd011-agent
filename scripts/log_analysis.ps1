$logContent = Get-Content "rd011_agent.log"
$startIndexArr = @()
for ($i = 0; $i -lt $logContent.Count; $i++) {
    if ($logContent[$i] -match "Using SQLite checkpointer") { $startIndexArr += $i }
}
if ($startIndexArr.Count -eq 0) { Write-Host "No start marker."; exit }
$selectedStarts = $startIndexArr | Select-Object -Last 2
foreach ($startIdx in $selectedStarts) {
    $idxInArr = [array]::IndexOf($startIndexArr, $startIdx)
    $nextStart = if ($idxInArr -lt $startIndexArr.Count - 1) { $startIndexArr[$idxInArr + 1] } else { $logContent.Count }
    $windowLines = $logContent[$startIdx..($nextStart - 1)]
    $stStr = ($windowLines[0] -split ' - ')[0]
    $enStr = ($windowLines[-1] -split ' - ')[0]
    Write-Host "`n=== WINDOW (Lines $($startIdx+1) to $nextStart) ==="
    Write-Host "Time: $stStr to $enStr"

    $markers = @(
        "ingest complete",
        "extraction complete",
        "planning complete",
        "issue detection complete",
        "approved",
        "intro generation complete",
        "render_diagrams",
        "assemble_document"
    )
    foreach ($m in $markers) {
        $found = $windowLines | Select-String $m | Select-Object -First 1
        if ($found) { Write-Host "$m: Line $($startIdx + $found.LineNumber) | $($found.ToString().Substring(0, 23))" }
    }

    $planned = ($windowLines | Select-String "Planning complete: (\d+) processes" | ForEach-Object { $_.Matches.Groups[1].Value } | Select-Object -Last 1)
    $genCount = ($windowLines | Select-String "generated section" | Measure-Object).Count
    Write-Host "Tasks: Planned=$planned, Generated=$genCount"

    $windowLines | Select-String "llm.router: Initialised \S+ for (\S+) \(model=(.+?)\)" |
        ForEach-Object { "$($_.Matches.Groups[1].Value): $($_.Matches.Groups[2].Value)" } |
        Sort-Object -Unique |
        ForEach-Object { Write-Host "Model: $_" }

    $att = ($windowLines | Select-String "LLM call attempt" | Measure-Object).Count
    $pas = ($windowLines | Select-String "Validation passed" | Measure-Object).Count
    $faW = ($windowLines | Select-String "Attempt \d+ failed with error" | Measure-Object).Count
    Write-Host "LLM: Attempts=$att, Passed=$pas, FailedWarn=$faW"

    $http = $windowLines | Select-String "https?://([^/\s]+).*?HTTP/1.1\\\" (\d{3})" |
        ForEach-Object { "$($_.Matches.Groups[1].Value) ($($_.Matches.Groups[2].Value))" } |
        Group-Object |
        ForEach-Object { "$($_.Count)x $($_.Name)" }
    Write-Host "HTTP Status: $($http -join ', ')"

    $rna = ($windowLines | Select-String "RAG not available" | Measure-Object).Count
    $rd = ($windowLines | Select-String "RAG disabled" | Measure-Object).Count
    $rqf = ($windowLines | Select-String "RAG query failed" | Measure-Object).Count
    $rex = $windowLines | Select-String "RAG exemplars selected for .* style_chars=(\d+) step_chars=(\d+)"
    $zst = 0
    if ($rex) { foreach ($re in $rex) { if ($re.Matches.Groups[2].Value -eq "0") { $zst++ } } }
    $che = ($windowLines | Select-String "Error querying Chroma" | Measure-Object).Count
    $rt0 = ($windowLines | Select-String "RAG_TRACE: results=0" | Measure-Object).Count
    $rtp = ($windowLines | Select-String "RAG_TRACE: results=[1-9]" | Measure-Object).Count
    Write-Host "RAG: NotAvail=$rna, Disabled=$rd, Fail=$rqf, Exemplars=$($rex.Count) (ZeroSteps=$zst), ChromaErr=$che, Trace0=$rt0, TracePos=$rtp"

    $gf = ($windowLines | Select-String "Generation failed for" | Measure-Object).Count
    $sw = ($windowLines | Select-String "Soft quality warnings" | Measure-Object).Count
    $hv = ($windowLines | Select-String "Hard quality violations" | Measure-Object).Count
    $sg = ($windowLines | Select-String "generated with soft quality warnings" | Measure-Object).Count
    $gtf = ($windowLines | Select-String "quality gate failed" | Measure-Object).Count
    Write-Host "Quality: GenFail=$gf, SoftWarn=$sw, HardViol=$hv, SoftGen=$sg, GateFail=$gtf"

    $dr = ($windowLines | Select-String "Rendered diagram" | Measure-Object).Count
    $mf = ($windowLines | Select-String "mmdc fallback" | Measure-Object).Count
    Write-Host "Diagrams: Render=$dr, MMDC=$mf"

    $errs = ($windowLines | Select-String "\[ERROR\]" | Measure-Object).Count
    $warns = ($windowLines | Select-String "\[WARNING\]" | Measure-Object).Count
    Write-Host "Totals: Errors=$errs, Warnings=$warns"
    if ($errs -gt 0) { $windowLines | Select-String "\[ERROR\]" | ForEach-Object { Write-Host "ERR: $($startIdx + $_.LineNumber): $($_.Line)" } }

    $success = ($windowLines -match "Workflow completed successfully" -or ($windowLines -match "assemble_document" -and $errs -eq 0))
    Write-Host "VERDICT: $(if ($success) { 'SUCCESS' } else { 'FAILURE' })"

    Write-Host "--- Top 10 Notable ---"
    $windowLines | Select-String "ERROR|WARNING|fail|timeout|RAG_TRACE" | Select-Object -First 10 | ForEach-Object {
        Write-Host "$($startIdx + $_.LineNumber): $($_.Line.Substring(0, [Math]::Min($_.Line.Length, 120)))"
    }
}
