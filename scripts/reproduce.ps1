param([string]$Scan = "benchmark/single_scan_with_ceiling", [string]$GroundTruth = "")
$py = ".\.venv\Scripts\python.exe"
& $py -m applied_ai.cli inspect --input $Scan
& $py -m applied_ai.cli run --input $Scan --tier lidar --config configs/lidar.yaml
$name = Split-Path $Scan -Leaf
& $py -m applied_ai.cli validate --scene outputs/$name/scene.json
if ($GroundTruth -ne "") { & $py -m applied_ai.cli evaluate --scene outputs/$name/scene.json --ground-truth $GroundTruth --calibrate }
