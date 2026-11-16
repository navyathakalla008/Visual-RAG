# VisualRAG / VPKE

VisualRAG / VPKE is a Windows-oriented Flask application for image understanding and video understanding. It combines a research image-text retrieval model with practical image analysis, video indexing, speech transcription, object and action detection, OCR, translation, and multimodal search.

The project provides two main experiences:

- **Image workspace:** upload an image, inspect its contents, captions, colors, shape, texture, pattern, pose/action, scene, and spatial characteristics.
- **Video workspace:** upload a video, analyze scenes and objects, generate speech transcripts, translate transcripts, and search speech, visuals, semantics, or all modalities together.

> This repository is an implementation and integration project inspired by the VisualRAG paper. It is not guaranteed to be an exact official reproduction. Internal implementation details that were not published by the paper should be validated experimentally.

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Complete File Reference](#complete-file-reference)
- [Installation on Windows](#installation-on-windows)
- [Preparing Flickr30K](#preparing-flickr30k)
- [Training](#training)
- [Evaluation](#evaluation)
- [Running the Flask Application](#running-the-flask-application)
- [Video Pipeline](#video-pipeline)
- [Configuration](#configuration)
- [API Routes](#api-routes)
- [Tests](#tests)
- [Runtime Directories and Generated Data](#runtime-directories-and-generated-data)
- [Known Limitations](#known-limitations)
- [Maintenance Notes](#maintenance-notes)

## Features

### Image understanding

- CLIP-based image and text embedding
- Text/caption retrieval
- YOLO object detection
- Grounded/open-vocabulary object detection support
- Pose and action detection
- Scene, material, semantic, color, shape, texture, and pattern analysis
- Spatial size/depth estimation
- Flickr30K entity and caption-assisted labeling
- DOM-style HTML image result interface

### VPKE image-text retrieval

- CLIP image encoding
- Auxiliary caption support
- Key-concept extraction
- Concept memory bank
- Dual-confidence concept selection
- Vision-guided semantic rebalancing
- Multi-scale hierarchical attention
- Dual-triplet retrieval training
- Image-to-text and text-to-image Recall@K evaluation

### Video understanding

- Scene cut and timeline detection
- Frame sampling and scene keyframes
- YOLO object tracking
- Face-assisted character portraits and identity grouping
- Character names found in speech transcripts or on-screen text, when scene evidence supports the match
- Pose and action detection
- CLIP-based visual scene understanding
- Whisper speech transcription with word timestamps
- Optional speaker diarization
- Audio-event detection
- Frame OCR
- Translation through NLLB
- Transcript, visual, semantic, and multimodal search
- Background processing and asynchronous status

## System Architecture

### Image retrieval pipeline

```text
Image
  -> CLIP image encoder
  -> VPKE image representation

Caption
  -> CLIP text encoder
  -> concept extraction
  -> concept memory bank
  -> dual-confidence concept selection
  -> vision-guided rebalancing
  -> VPKE text representation

Image representation <- cosine similarity -> Text representation
                         |
                         -> dual triplet retrieval loss
```

### Video pipeline

```text
Uploaded video
  -> directory and index setup
  -> frame sampling
  -> scene detection and timeline validation
  -> object detection/tracking
  -> pose/action detection
  -> scene classification
  -> OCR
  -> audio extraction
  -> Whisper transcription
  -> optional translation/diarization/audio events
  -> JSON and embedding index
  -> phrase/visual/semantic/multimodal search
```

## Project Structure

```text
VisualRAG_VPKE_code/
├── app.py                          # Main Flask application
├── video_pipeline.py               # Video analysis and indexing
├── video_search.py                 # Video search implementations
├── video_ui.py                     # Video web interface
├── speech_transcribe.py            # Whisper transcription helpers
├── diarization_service.py          # Optional speaker diarization
├── ocr_service.py                  # Frame OCR
├── translation_service.py          # Transcript translation and cache
├── audio_events.py                 # Audio-event detection
├── model_registry.py               # Optional model registration
├── config.py                       # Default research configuration
├── train.py                        # VPKE training entry point
├── evaluate.py                     # VPKE retrieval evaluation
├── losses.py                       # Dual-triplet loss
├── metrics.py                      # Recall and retrieval metrics
├── plot_results.py                 # Result visualization
├── data/
│   └── flickr30k.py                # Flickr30K dataset loader
├── features/
│   └── visual_features.py          # Reserved visual feature module
├── models/
│   ├── vpke.py                     # Complete VPKE assembly
│   ├── clip_encoder.py             # CLIP wrapper
│   ├── knowledge.py                # Concepts and dual-confidence selection
│   ├── rebalancing.py              # Semantic rebalancing
│   ├── msha.py                     # Multi-scale hierarchical attention
│   └── blip2_caption.py            # Optional caption generation
├── tests/                          # Test entry points are currently root-level
├── uploads/                        # Uploaded images, videos, indexes, and cache
├── checkpoints/                    # Saved training checkpoints
├── weights/                        # Optional local model weights
└── requirements.txt                # Python dependencies
```

## Complete File Reference

### Core application files

| File | Purpose |
|---|---|
| `app.py` | Main Flask server. Handles static-image analysis, image upload, YOLO/CLIP inference, visual feature extraction, and video routes. |
| `video_pipeline.py` | Samples frames, detects scenes, tracks objects and actions, runs scene understanding/OCR/transcription, validates timelines, and saves video indexes. |
| `video_search.py` | Implements exact phrase, visual concept, semantic, and multimodal video search. |
| `video_ui.py` | Contains the HTML/CSS/JavaScript for the video interface. |
| `videomae_action_worker.py` | Runs VideoMAE action detection in a separate Python environment. |

### Model and retrieval files

| File | Purpose |
|---|---|
| `models/vpke.py` | Assembles CLIP, concept selection, rebalancing, and hierarchical attention modules. |
| `models/clip_encoder.py` | Loads and wraps an OpenCLIP image/text encoder. |
| `models/knowledge.py` | Extracts concepts, builds/loads a concept memory bank, and performs dual-confidence concept selection. |
| `models/rebalancing.py` | Rebalances original text, auxiliary captions, and concepts according to image similarity. |
| `models/msha.py` | Implements multi-scale convolution/attention layers and hierarchical feature projection. |
| `models/blip2_caption.py` | Optional BLIP-style auxiliary-caption helper. |
| `models/__init__.py` | Marks the `models` directory as a Python package and exports package initialization. |
| `losses.py` | Implements image-to-text and text-to-image triplet losses and weighted dual-triplet loss. |
| `metrics.py` | Computes Recall@1, Recall@5, Recall@10 and rSum retrieval metrics. |
| `config.py` | Central data class containing the default model, retrieval, training, and loss settings. |
| `model_registry.py` | Centralizes optional GPU-aware model selection and environment-based feature enablement. |

### Image-analysis files

| File | Purpose |
|---|---|
| `build_concept_features.py` | Builds concept features for retrieval or image analysis. |
| `extract_concepts.py` | Extracts NLP concepts from captions or text. |
| `generate_auxiliary_captions.py` | Generates auxiliary textual descriptions for images. |
| `concept_features.pt` | Generated torch concept-feature tensor. Runtime/generated artifact. |
| `concept_memory.txt` | Generated list of known concepts. Runtime/generated artifact. |
| `image_concepts.py` | Image-specific concept processing helpers. |
| `image_detector.py` | Image/object detection helpers used by the application. |
| `flickr_entities.py` | Flickr entity/caption vocabulary and entity-label helpers. |
| `object_filter.py` | Filters duplicate or weak object detections. |
| `features/visual_features.py` | Reserved/experimental visual-feature module; currently empty. |
| `features/` | Package directory for visual feature extraction extensions. |

### Video and multimodal-service files

| File | Purpose |
|---|---|
| `audio_events.py` | Extracts audio and classifies overlapping 5-second windows with the 527-label AudioSet AST model using multi-label sigmoid scores. Event boundaries are approximate model windows, not frame-accurate ground truth. |
| `diarization_service.py` | Optional speaker-separation/diarization integration. |
| `ocr_service.py` | Reads text from selected video frames and provides availability checks. |
| `speech_transcribe.py` | Finds FFmpeg, extracts audio, loads Whisper, detects language, and returns timestamped transcription. |
| `translation_service.py` | Translates transcripts with NLLB and caches translation results. |
| `update_index_clean.py` | Maintenance script for cleaning or rebuilding saved video indexes. |
| `video_pipeline_backup.py` | Older video-pipeline backup. |
| `video_pipeline_backup_before_fast_preview.py` | Older pipeline backup taken before fast-preview changes. |
| `video_ui_before_video_fix.py` | Older video UI backup. |
| `video_ui.py.before_video_patch_20261026_180219.bak` | Timestamped UI backup. |
| `app_backup_before_fast_preview.py` | Full Flask-app backup before fast-preview changes. |
| `app_before_video_fix.py` | Flask-app backup before a video repair. |
| `app.py.before_video_fix.bak` | Timestamped Flask-app backup. |
| `app.py.before_video_patch_20261026_180219.bak` | Timestamped Flask-app backup. |
| `_recovered_video_ui.py` | Recovered copy of an earlier video UI. |
| `_job_html.txt` | Saved generated job/HTML output fragment. |
| `_job_js.txt` | Saved generated job JavaScript fragment. |

### Data and training files

| File | Purpose |
|---|---|
| `data/flickr30k.py` | Loads Flickr30K captions, images, and official split lists into a PyTorch dataset. |
| `data/__init__.py` | Marks `data` as a package. |
| `train.py` | CLI entry point for training VPKE with AdamW, dual-triplet loss, batching, and checkpoint saving. |
| `evaluate.py` | CLI entry point for loading a checkpoint and evaluating bidirectional Recall metrics. |
| `plot_results.py` | Produces plots for retrieval or experiment results. |

### Tests

| File | Purpose |
|---|---|
| `test_vpke.py` | Unit test for VPKE forward shapes and configuration plumbing. |
| `test_video_page_render.py` | Checks rendering behavior for the video page UI. |
| `test_scene_search_fixes.py` | Tests timeline normalization and phrase/semantic search behavior. |
| `test_videomae.py` | Tests isolated VideoMAE action-worker integration. |
| `test_e2e_real_video.py` | Runs an end-to-end real-video validation when suitable input is available. |
| `verify_crops.py` | Diagnostic script for checking detection crops. |
| `verify_objects.py` | Diagnostic script for object detection results. |
| `verify_objects_v2.py` | Alternate diagnostic script for object detection verification. |
| `yolo_test.jpg` | Small image used by YOLO/vision verification scripts. |
| `temp_1000092795.jpg` | Temporary image artifact saved during local testing. |

### Environment and setup files

| File | Purpose |
|---|---|
| `requirements.txt` | Primary Python dependencies. |
| `fix_visualrag_video.ps1` | Windows repair script for video functionality and model environment setup. |
| `run_quick_test.bat` | Convenience batch file for running a quick project check. |
| `INSTALL_FIX.txt` | Installation/repair notes. |
| `INSTALL_README.txt` | Windows installation notes for the combined repair package. |
| `README.md` | Main technical project documentation, setup, and file reference. |
| `README.txt` | Short notes about reliable video upload and background processing. |
| `README_FIX_NOW.txt` | Quick repair instructions for VideoMAE integration. |
| `README_WORD_FREQUENCY_FIX.txt` | Notes for transcript word and phrase frequency fixes. |

### Model weights and generated artifacts

| File or directory | Purpose |
|---|---|
| `yolo11n.pt` | YOLO11 nano detection weights. |
| `yolo11n-pose.pt` | YOLO pose-estimation weights. |
| `yolov8s-world.pt` | YOLO World open-vocabulary detection weights. |
| `weights/clip/ViT-B-32.pt` | Optional local CLIP weights. |
| `checkpoints/last.pt` | Most recently generated VPKE training checkpoint. |
| `venv/` | Main local Windows Python virtual environment. Normally generated locally and should not be committed. |
| `videomae_test/` | Separate environment used to execute VideoMAE without dependency conflicts. Normally generated locally and should not be committed. |

Backup files should not be used by the running application. They are retained for recovery and comparison with earlier versions.

## Installation on Windows

### 1. Create the main environment

```powershell
cd C:\Users\HP\Downloads\VisualRAG_VPKE_code
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Model libraries may download weights when first used.

### 2. Install optional VideoMAE environment

```powershell
py -3.11 -m venv videomae_test
videomae_test\Scripts\activate
python -m pip install torch torchvision transformers==4.57.1 opencv-python Pillow
```

The VideoMAE worker is invoked through:

```text
videomae_test\Scripts\python.exe
```

### 3. Download/model availability

Some components require internet access on first use, including Whisper, NLLB, Hugging Face models, OpenCLIP weights, and optional diarization dependencies. The application falls back where implemented, but complete video analysis requires FFmpeg and enough local CPU/GPU/RAM.

## Preparing Flickr30K

The dataset loader requires the official image and split layout:

```text
Flickr30K/
├── Images/
├── captions.txt
├── train.txt
├── val.txt
└── test.txt
```

`captions.txt` is read as CSV with columns named:

```text
image,caption
1000092795.jpg,Two people are walking beside water.
```

Each split file should contain one image filename or stem per line:

```text
1000092795.jpg
1000245624.jpg
```

The loader also adds `.jpg` automatically if a split line has no extension.

## Training

Train with a small smoke-test run:

```powershell
venv\Scripts\activate
python train.py --data_root C:\datasets\Flickr30K --epochs 1 --batch_size 8 --max_train_samples 1000
```

Full training:

```powershell
python train.py `
  --data_root C:\datasets\Flickr30K `
  --epochs 20 `
  --batch_size 64 `
  --workers 2 `
  --lr 0.00003 `
  --output_dir checkpoints
```

The saved checkpoint is written to:

```text
checkpoints\last.pt
```

## Evaluation

```powershell
python evaluate.py `
  --checkpoint checkpoints\last.pt `
  --data_root C:\datasets\Flickr30K `
  --batch_size 128
```

Evaluation reports:

- Image-to-text Recall@1/@5/@10
- Text-to-image Recall@1/@5/@10
- Mean recall
- rSum

## Running the Flask Application

```powershell
cd C:\Users\HP\Downloads\VisualRAG_VPKE_code
venv\Scripts\activate
python app.py
```

Open:

```text
http://127.0.0.1:5000/
http://127.0.0.1:5000/video
```

For a production deployment, use a WSGI server rather than Flask's development server. Add authentication before exposing the application to the network.

## Video Pipeline

The video pipeline supports:

- `ffmpeg` and `ffprobe`
- Frame sampling
- Scene-cut detection
- Scene timeline validation
- YOLO object detection/tracking
- Pose-estimation models
- CLIP scene/action classification
- VideoMAE temporal action classification through an isolated worker
- Whisper transcription
- Word-level timestamps
- Speaker diarization
- OCR
- Translation
- Audio-event analysis
- Multilingual search normalization
- Cached indexes and scene embeddings

Upload processing runs in the background. The job status endpoint returns progress so the UI can update without blocking.

### Video search modes

| Mode | Behavior |
|---|---|
| `phrase` | Finds exact repeated transcript phrases and returns timestamps. |
| `visual` | Searches objects, actions, relationships, scene concepts, and visible labels. |
| `semantic` | Uses transcript/scene text and embeddings or CLIP-based relevance. |
| `multimodal` | Combines visual concepts, transcript text, OCR, and scene evidence. |

## Configuration

The default VPKE settings are defined in `config.py`:

| Setting | Default |
|---|---|
| CLIP model | `ViT-B-16` |
| Feature dimension | 512 |
| Similarity dimension | 256 |
| Attention heads | 4 |
| Hierarchical layers | 6 |
| Selected concepts (`Gk`) | 10 |
| Margin | 0.5 |
| Fused-triplet weight (`mu`) | 8.0 |
| Entity-triplet weight (`eta`) | 1.0 |
| Training epochs | 20 |

Environment variables are also used by `model_registry.py`, including:

```text
WHISPER_MODEL
OBJECT_DETECTION_MODEL
GROUNDING_DINO_ENABLED
SAM2_ENABLED
ACTION_DETECTION_MODEL
AUDIO_EVENT_MODEL
CLAP_ENABLED
SEMANTIC_EMBEDDING_MODEL
BGE_M3_ENABLED
VIDEO_REASONING_MODEL_ENABLED
```

Optional Hugging Face diarization credentials must be supplied through the service environment/configuration used by `diarization_service.py`.

## API Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET, POST | Static-image upload and analysis page. |
| `/uploaded/<filename>` | GET | Serves an uploaded image. |
| `/video` | GET, POST | Video upload page and video analysis submission. |
| `/video/job/<job_id>` | GET | Reads background analysis status. |
| `/video/transcribe` | POST | Starts speech transcription for an analyzed video. |
| `/video/translate` | POST | Translates a video's transcript into a supported language. |
| `/video/search` | POST | Searches a video index by phrase, visual, semantic, or multimodal mode. |
| `/uploaded-video/<filename>` | GET | Serves uploaded videos. |
| `/video-frame/<video_id>/<filename>` | GET | Serves extracted scene frames. |

The Flask application is stateful: indexes, translations, and uploaded files remain on disk.

## Tests

Run syntax compilation:

```powershell
venv\Scripts\activate
python -m compileall app.py video_pipeline.py video_search.py video_ui.py speech_transcribe.py
```

Run available focused scripts:

```powershell
python test_vpke.py
python test_scene_search_fixes.py
python test_video_page_render.py
python test_videomae.py
```

The end-to-end script may require existing local videos and should only be run when its required fixtures are available:

```powershell
python test_e2e_real_video.py
```

Object-detection checks are diagnostic utilities:

```powershell
python verify_crops.py
python verify_objects.py
python verify_objects_v2.py
```

## Runtime Directories and Generated Data

### `uploads/`

This directory contains runtime data and should generally be excluded from source control:

```text
uploads/
├── <image filename>                  Uploaded/static test images
├── videos/                           Stored uploaded video files
├── video_frames/                     Extracted frames and scene keyframes
├── video_index/                      JSON indexes, WAV files, and embedding tensors
└── translation_cache/                Cached translated transcripts
```

Video indexes contain scene boundaries, transcript segments, word timings, object tracks, actions, detected concepts, audio/OCR data, and provenance fields where available.

### `checkpoints/`

Stores PyTorch training checkpoints. `checkpoints/last.pt` contains the latest model state and epoch value.

### `weights/`

Stores optional local model weights. Heavy weight files and downloads should normally be excluded from Git.

### Virtual environments

`venv/` and `videomae_test/` are local environments. They contain installed packages and executable interpreters and should not be committed or documented line-by-line as application source.

## Known Limitations

- The implementation uses a practical approximation of the research paper; not every published low-level detail may be represented.
- Official Flickr30K split files are required for a proper experiment.
- Auxiliary-caption generation and concept-memory quality affect retrieval results.
- Search quality depends heavily on local model/device availability.
- Video scene boundaries may differ across codecs and frame rates.
- Whisper transcription requires audible speech; silence or music can produce no transcript.
- VideoMAE is trained for broad Kinetics-400 action classes and is not a universal action recognizer.
- YOLO and pose detections can produce false positives or miss small/occluded objects.
- Character matching uses local YuNet and SFace models with CLIP fallback; small, stylized, or occluded faces can still be missed or mismatched.
- Character names are extracted with spaCy's English `en_core_web_sm` model from transcript and OCR text. Install the model with `python -m spacy download en_core_web_sm`; a name is displayed only when scene evidence supports a single-character match.
- Translation quality varies by language, domain, and model size.
- Fast image and video models require significant CPU/GPU memory.
- The current test suite includes diagnostic and end-to-end scripts and does not cover every code path.
- Runtime uploads, indexes, and cache data may contain user content and must not be shared or committed.

## Maintenance Notes

- Do not delete backup files before confirming that the active `app.py`, `video_pipeline.py`, and `video_ui.py` are correct.
- Regenerate a video index if the pipeline version, scene analysis, or search behavior changes.
- Re-run `python -m compileall` after merging backup or repair files.
- Keep Python/model dependencies pinned when producing reproducible academic results.
- Report expected hardware, FFmpeg version, Python version, CUDA/PyTorch versions, and model names in experiment results.
- Do not claim “first-ever” or “no one has done this” novelty without checking recent papers, patents, and comparable implementations.
