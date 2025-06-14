# PowerShell-Skript zum Starten mehrerer Trainingsläufe mit verschiedenen Parametern

$metrics         = @("Margin", "Entropy", "LeastConfidence")
$accuracies      = @(50, 55, 60, 65, 70, 75, 80, 85, 90)
$boot_epochs     = 10
$epochs          = 0
$cycle           = 0
$n_query         = 200
$n_split         = 10
$workers         = 4
$batch_size      = 256
$test_batch_size = 500

$dataset   = "FashionMNIST"
$model     = "ResNet18"
$gpu_index = 0
$optimizer = "SGD"
$lr        = 0.1
$momentum  = 0.9
$scheduler = "CosineAnnealingLR"

$logDir = "logs"
$csv_log = "$logDir\summary_log.csv"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

if (-not (Test-Path $csv_log)) {
    # Header schreiben
    "metric,target_accuracy,boot_epochs,duration_sec,accuracy,precision,recall,f1,log_file" `
        | Out-File -FilePath $csv_log
}

foreach ($metric in $metrics) {
    foreach ($targetAcc in $accuracies) {
        # Kommando zusammensetzen
        $cmd = @(
            "python -m src.play_it_stright.main_play_it_straight",
            "--dataset $dataset",
            "--model $model",
            "--gpu $gpu_index",
            "--batch-size $batch_size",
            "--test-batch-size $test_batch_size",
            "--boot_epochs $boot_epochs",
            "--epochs $epochs",
            "--cycle $cycle",
            "--method Uncertainty",
            "--n-query $n_query",
            "--n_split $n_split",
            "--workers $workers",
            "--optimizer $optimizer",
            "--lr $lr",
            "--momentum $momentum",
            "--scheduler $scheduler",
            "--uncertainty $metric",
            "--target_accuracy $targetAcc",
            "--resume False",
            "--print_freq 1"
        ) -join " "

        $log_file = "$logDir\${metric}_${targetAcc}.log"
        Write-Host "`n==== Running: ${metric} | Target: ${targetAcc}% | Boot-Epochs: ${boot_epochs} ====" -ForegroundColor Cyan

        $startTime = Get-Date
        Invoke-Expression $cmd | Tee-Object -FilePath $log_file -Append
        $endTime   = Get-Date
        $duration  = ($endTime - $startTime).TotalSeconds

        # Boot-Ergebnis parsen
        $bootLine = Select-String -Path $log_file -Pattern 'Boot completed \| Accuracy: (?<acc>\d+\.\d+), Precision: (?<prec>\d+\.\d+), Recall: (?<rec>\d+\.\d+), F1: (?<f1>\d+\.\d+)'
        if ($bootLine) {
            $bootAcc = [double]$bootLine.Matches.Groups['acc'].Value
            $prec    = [double]$bootLine.Matches.Groups['prec'].Value
            $rec     = [double]$bootLine.Matches.Groups['rec'].Value
            $f1      = [double]$bootLine.Matches.Groups['f1'].Value
        } else {
            # Fallback
            $bootAcc = '' 
            $prec    = ''
            $rec     = ''
            $f1      = ''
        }

        # In CSV schreiben
        "$metric,$targetAcc,$boot_epochs,$duration,$bootAcc,$prec,$rec,$f1,$log_file" `
            | Out-File -FilePath $csv_log -Append
    }
}
