$logFile = "rd011_agent.log"
$lines = Get-Content $logFile
$startIndices = @()
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match "Using SQLite checkpointer") {
        $startIndices += $i
    }
}
$windowCount = $startIndices.Count
if ($windowCount -gt 0) {
    $numToTake = if ($windowCount -ge 2) { 2 } else { $windowCount }
    $lastTwo = $startIndices | Select-Object -Last $numToTake
    foreach ($startIndex in $lastTwo) {
        $windowIdx = [array]::IndexOf($startIndices, $startIndex)
        $endIndex = if ($windowIdx -lt $windowCount - 1) { $startIndices[$windowIdx + 1] - 1 } else { $lines.Count - 1 }
        $windowLines = $lines[$startIndex..$endIndex]

        $startTimeStr = ($windowLines[0] -split ' - ')[0]
        $endTimeStr = ($windowLines[-1] -split ' - ')[0]
        $duration = "N/A"
        try {
            $st = [DateTime]::ParseExact($startTimeStr, "yyyy-MM-dd HH:mm:ss,fff", $null)
            $et = [DateTime]::ParseExact($endTimeStr, "yyyy-MM-dd HH:mm:ss,fff", $null)
            $duration = ($et - $st).TotalSeconds
        } catch {}

        Write-Host "`n===================================================="
        Write-Host "WINDOW: Lines $($startIndex + 1) to $($endIndex + 1)"
        Write-Host "Time: $startTimeStr to $endTimeStr ($duration s)"

        $stageList = @(
            ('Ingest', 'ingest complete'),
            ('Extract', 'extraction complete'),
            ('Plan', 'planning complete'),
            ('IssueDet', 'issue detection complete'),
            ('Approval', 'approved'),
            ('Intro', 'intro generation complete'),
            ('Render', 'render_diagrams'),
            ('Assemble', 'assemble_document')
        )
        foreach ($s in $stageList) {
            $m = $windowLines | Select-String $s[1] | Select-Object -First 1
            if ($m) { Write-Host "$($s[0]): Line $($startIndex + $m.LineNumber) | $($m.ToString() -split ' - ')[0]" }
        }

        $sectGen = $windowLines | Select-String "Generating content for section"
        if ($sectGen) {
            Write-Host "FirstSectStart: Line $($startIndex + $sectGen[0].LineNumber) | $($sectGen[0].ToString() -split ' - ')[0]"
            Write-Host "LastSectGen: Line $($startIndex + $sectGen[-1].LineNumber) | $($sectGen[-1].ToString() -split ' - ')[0]"
        }

        $plannedProcess = 0
        $ppMatch = $windowLines | Select-String "Planning complete: (\d+) processes"
        if ($ppMatch) { $plannedProcess = $ppMatch[-1].Matches.Groups[1].Value }
        $genSections = ($windowLines | Select-String "generated section" | Measure-Object).Count
        Write-Host "Planned Processes: $plannedProcess | Generated Sections: $genSections"

        $windowLines | Select-String "llm.router: Initialised (\S+) for (\S+) \(model=(.+?)\)" |
            ForEach-Object { "$($_.Matches.Groups[2].Value): $($_.Matches.Groups[3].Value)" } |
            Sort-Object -Unique |
            ForEach-Object { Write-Host "ModelInit: $_" }

        $att = ($windowLines | Select-String "LLM call attempt" | Measure-Object).Count
        $pas = ($windowLines | Select-String "Validation passed" | Measure-Object).Count
        $faW = ($windowLines | Select-String "Attempt \d+ failed with error" | Measure-Object).Count
        Write-Host "LLM Metrics - Att: $att | Pas: $pas | FailW: $faW"

        $windowLines | Select-String "https?://([^/\s]+).*?HTTP/1.1\\\" (\d{3})" |
            ForEach-Object { "$($_.Matches.Groups[1].Value) | $($_.Matches.Groups[2].Value)" } |
            Group-Object |
            ForEach-Object { Write-Host "HTTP: $($_.Count) x $($_.Name)" }

        $rna = ($windowLines | Select-String "RAG not available" | Measure-Object).Count
        $rd = ($windowLines | Select-String "RAG disabled" | Measure-Object).Count
        $rqf = ($windowLines | Select-String "RAG query failed" | Measure-Object).Count
        $rex = $windowLines | Select-String "RAG exemplars selected for .* style_chars=(\d+) step_chars=(\d+)"
        $zst = 0
        if ($rex) { foreach ($re in $rex) { if ($re.Matches.Groups[2].Value -eq "0") { $zst++ } } }
        $che = ($windowLines | Select-String "Error querying Chroma" | Measure-Object).Count
        $rt0 = ($windowLines | Select-String "RAG_TRACE: results=0" | Measure-Object).Count
        $rtp = ($windowLines | Select-String "RAG_TRACE: results=[1-9]" | Measure-Object).Count
        Write-Host "RAG - NotAvail: $rna | Disabled: $rd | QueryFail: $rqf | Exemplars: $($rex.Count) (Zero: $zst) | ChromaErr: $che | Trace0: $rt0 | TraceP: $rtp"

        $gf = ($windowLines | Select-String "Generation failed for" | Measure-Object).Count
        $sw = ($windowLines | Select-String "Soft quality warnings" | Measure-Object).Count
        $hv = ($windowLines | Select-String "Hard quality violations" | Measure-Object).Count
        $sg = ($windowLines | Select-String "generated with soft quality warnings" | Measure-Object).Count
        $gtf = ($windowLines | Select-String "quality gate failed" | Measure-Object).Count
        Write-Host "Quality - GenFail: $gf | SoftW: $sw | HardV: $hv | SoftG: $sg | GateF: $gtf"

        $dr = ($windowLines | Select-String "Rendered diagram" | Measure-Object).Count
        $mf = ($windowLines | Select-String "mmdc fallback" | Measure-Object).Count
        Write-Host "Diagrams - Render: $dr | MMDC: $mf"

        $ert = ($windowLines | Select-String "\[ERROR\]" | Measure-Object).Count
        $wat = ($windowLines | Select-String "\[WARNING\]" | Measure-Object).Count
        Write-Host "Errors: $ert | Warnings: $wat"
        if ($ert -gt 0) {
            $windowLines | Select-String "\[ERROR\]" | ForEach-Object {
                Write-Host "ERR: $($startIndex + $_.LineNumber): $($_.Line)"
            }
        }

        $success = ($windowLines -match "Workflow completed successfully" -or ($windowLines -match "assemble_document" -and $ert -eq 0))
        Write-Host "VERDICT: $(if ($success) { 'SUCCESS' } else { 'FAILURE' })"

        $windowLines |
            Select-String "ERROR|WARNING|fail|exception|retry|timeout|abort|RAG_TRACE|critical|invalid" |
            Select-Object -First 10 |
            ForEach-Object {
                $tr = if ($_.Line.Length -gt 120) { $_.Line.Substring(0, 117) + "..." } else { $_.Line }
                Write-Host "Note: $($startIndex + $_.LineNumber): $tr"
            }
    }
}
