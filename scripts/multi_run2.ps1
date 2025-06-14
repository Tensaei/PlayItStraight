# PowerShell-Skript zum Starten mehrerer Trainingsläufe mit verschiedenen Parametern

$metrics = @("Margin", "Entropy", "LeastConfidence")
$accuracies = @(50, 55, 60, 65, 70, 75, 80, 85, 90)

# Logs-Ordner erstellen, falls nicht vorhanden
$logDir = "logs"
$csv_log = "$logDir\summary_log.csv"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

if (-not (Test-Path $csv_log)) {
    "metric,target_accuracy,boot_epochs,duration_sec,log_file" | Out-File -FilePath $csv_log
}


foreach ($metric in $metrics) {
    foreach ($acc in $accuracies) {
        $cmd = "python -m src.play_it_stright.main_play_it_straight --dataset FashionMNIST --model ResNet18 --gpu 0 --batch-size 128 --test-batch-size 500 --boot_epochs 10 --epochs 5 --cycle 5 --method Uncertainty --n-query 200 --n_split 10 --workers 4 --optimizer SGD --lr 0.1 --momentum 0.9 --scheduler CosineAnnealingLR --uncertainty $metric --target_accuracy $acc --resume False --print_freq 1"
        
        $log_file = "$logDir\$metric`_$acc.log"
        Write-Host "`n==== Running: $metric | Target: $acc% | Boot-Epochs: 5 ====" -ForegroundColor Cyan

        $startTime = Get-Date
        Invoke-Expression $cmd | Tee-Object -FilePath $log_file -Append
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds

        "$metric,$acc,$boot_epochs,$duration,$log_file" | Out-File -FilePath $csv_log -Append
    }
}

