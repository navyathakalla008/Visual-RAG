VisualRAG quick repair (Windows)

Files:
- video_pipeline.py: updated pipeline (VideoMAE temporal actions + FFmpeg path fix for Whisper)
- videomae_action_worker.py: isolated VideoMAE inference worker

Install:
1. Stop the Flask app with Ctrl+C.
2. Back up the existing project video_pipeline.py.
3. Copy the included video_pipeline.py and videomae_action_worker.py into:
   C:\Users\HP\Downloads\VisualRAG_VPKE_code
4. Keep the existing app.py and original venv unchanged.
5. In PowerShell, from the project folder run:
   venv\Scripts\activate
   python -m pip show imageio-ffmpeg openai-whisper
   python app.py

The separate VideoMAE environment must already exist at:
   .\videomae_test\Scripts\python.exe
and must contain torch, torchvision, transformers 4.57.1, opencv-python and Pillow.

Important limits:
- This is a practical improvement, not a guarantee of perfect predictions for every possible video.
- VideoMAE Kinetics-400 predicts one broad class per short clip; it can miss actions/classes outside its training labels.
- Speech transcript requires audible speech. Music/noise/silent clips may correctly produce no transcript.
- On first run, model download may take time and RAM is limited; don't run multiple analyses simultaneously.
- PIPELINE_VERSION was increased to 10 so stale version-9 indexes should be rebuilt.
