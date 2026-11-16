VisualRAG Step 1: reliable upload and background processing

1. Stop the current Flask server with Ctrl+C.
2. Back up app.py, video_ui.py, and video_pipeline.py in your project.
3. Copy these three files into C:\Users\HP\Downloads\VisualRAG_VPKE_code, replacing the old files.
4. Run: python app.py
5. Open http://127.0.0.1:5000/video and upload a video.

This step fixes the confirmed _normalize_scene_timeline(progress=...) crash, avoids reusing an old index for a new upload, keeps live status under the player rather than a separate Processing video card, and prints full tracebacks if processing fails.

This package has syntax checks only; it has not been run against your local models/video. If another error occurs, send the full terminal traceback.
