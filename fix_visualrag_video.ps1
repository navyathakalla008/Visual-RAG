# Run from C:\Users\HP\Downloads\VisualRAG_VPKE_code
# This script makes timestamped backups and patches only the targeted code blocks.
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\app.py") -or -not (Test-Path ".\video_ui.py")) {
    throw "Run this script from the VisualRAG_VPKE_code project folder."
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item ".\app.py" ".\app.py.before_video_patch_$stamp.bak"
Copy-Item ".\video_ui.py" ".\video_ui.py.before_video_patch_$stamp.bak"

# Patch app.py: set visible status before model initialization, and recover the
# stored filename from the video ID if job metadata lacks filename.
$app = Get-Content ".\app.py" -Raw

$oldWorker = @'
def _run_video_job(job_id, video_path, video_id, stored_name, original_name):
    def progress(message):
        _set_job(job_id, status="running", message=message)

    try:
        with _video_analyze_lock:
'@
$newWorker = @'
def _run_video_job(job_id, video_path, video_id, stored_name, original_name):
    def progress(message):
        _set_job(job_id, status="running", message=message)

    # Mark the job as running before model initialization or lock waiting.
    _set_job(
        job_id,
        status="running",
        message="Starting video analysis and loading vision models...",
        video_id=video_id,
        filename=stored_name,
    )
    try:
        with _video_analyze_lock:
'@
if (-not $app.Contains($oldWorker)) {
    throw "Could not find the expected _run_video_job block. No app.py changes were written."
}
$app = $app.Replace($oldWorker, $newWorker)

$oldFilenameLine = '        uploaded_filename = job.get("filename") if job else None'
$newFilenameLine = @'
        uploaded_filename = job.get("filename") if job else None

        # Recover the uploaded filename from the video ID if job metadata is incomplete.
        if job and not uploaded_filename:
            job_video_id = job.get("video_id")
            if job_video_id:
                try:
                    for candidate in os.listdir(VIDEO_UPLOAD_DIR):
                        if candidate.startswith(job_video_id + "_"):
                            uploaded_filename = candidate
                            _set_job(job_id, filename=candidate)
                            break
                except OSError as exc:
                    print("Could not inspect video upload directory:", exc)
'@
if (-not $app.Contains($oldFilenameLine)) {
    throw "Could not find the expected uploaded_filename line. No app.py changes were written."
}
$app = $app.Replace($oldFilenameLine, $newFilenameLine)
Set-Content ".\app.py" -Value $app -Encoding UTF8

Write-Host "Patched app.py. video_ui.py already contains a conditional player; the server-side filename recovery above makes it render when metadata is incomplete." -ForegroundColor Green
Write-Host ""
Write-Host "Next: stop Flask with Ctrl+C, then run: python app.py"
Write-Host "If the old job is still queued after restart, upload the video again to create a fresh job."
