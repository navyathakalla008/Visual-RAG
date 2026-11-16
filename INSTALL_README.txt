VisualRAG Combined Repair (Windows)

Included files:
- app.py: your uploaded app version, retaining /video/transcribe and /video/search routes
- video_pipeline.py: FFmpeg lookup support, Whisper transcript handling, VideoMAE temporal action worker integration, cache pipeline version 10
- videomae_action_worker.py: separate-environment VideoMAE worker
- video_search.py: visual/semantic/speech search and word/phrase occurrence counting
- video_ui.py: search UI and word/phrase occurrence results

INSTALL:
1. Stop Flask with Ctrl+C.
2. Back up your current app.py, video_pipeline.py, video_search.py, and video_ui.py.
3. Copy the five included files into C:\Users\HP\Downloads\VisualRAG_VPKE_code, replacing those files.
4. Keep your existing models/, flickr_entities.py, other project files, .env/configuration, uploads, and data folders.
5. Activate the original environment and install/verify dependencies:
   venv\Scripts\activate
   python -m pip install imageio-ffmpeg openai-whisper
6. Confirm the separate VideoMAE environment exists at:
   videomae_test\Scripts\python.exe
   and contains torch, torchvision, transformers==4.57.1, opencv-python, Pillow.
7. Run:
   python -m py_compile app.py video_pipeline.py video_search.py video_ui.py videomae_action_worker.py
   python app.py

IMPORTANT:
- The included app.py is the version uploaded in this chat. If you have made newer local changes after uploading it, compare/merge before replacing.
- The package is syntax-checked only in this environment; it has not been run against your local Windows model caches or data folders.
- Speech transcription requires Whisper's model download to be available and enough time/RAM. For Telugu, the transcription path must use multilingual Whisper model (e.g. small or medium), not an English-only .en model.
- VideoMAE Kinetics-400 is a limited action classifier, not a universal video understanding model. No package can guarantee correct predictions for every video. Validate output on your videos.
- Re-analysis with pipeline version 10 prevents reusing older pipeline indexes when find_matching_index sees the old version; existing indexes can still be opened directly and may need re-upload/re-analysis.
- Word/phrase frequency searches the saved transcript. If transcription is empty, run Transcribe first.
