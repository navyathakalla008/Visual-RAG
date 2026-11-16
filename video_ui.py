VIDEO_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>VisualRAG - Video Understanding</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* { box-sizing: border-box; }
body {
    font-family: "Segoe UI", Calibri, system-ui, sans-serif;
    background: #eef3f8;
    color: #132033;
    margin: 0;
    padding: 28px 4vw 56px;
}
.container { width: min(1440px, 100%); margin: 0 auto; }
.header { text-align: center; margin-bottom: 18px; }
.header h1 { margin: 0; font-size: 34px; color: #163a5c; letter-spacing: -0.5px; }
.header p { margin-top: 10px; color: #64748b; font-size: 15px; }
.nav-tabs {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-bottom: 24px;
}
.nav-tab {
    text-decoration: none;
    color: #1f4e79;
    background: white;
    border: 1px solid #dbe4ee;
    border-radius: 999px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 14px;
}
.nav-tab.active {
    background: #1f4e79;
    color: white;
    border-color: #1f4e79;
}
.card {
    background: white;
    padding: 28px;
    margin-bottom: 24px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
}
.collapsible-card { padding: 0; overflow: hidden; margin-bottom: 16px; }
.section-toggle {
    list-style: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 20px 24px;
}
.section-toggle::-webkit-details-marker { display: none; }
.section-toggle h2 { margin: 0; }
.section-toggle::after {
    content: "⌄";
    flex: 0 0 auto;
    width: 32px;
    height: 32px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 1px solid #dbe5ef;
    border-radius: 50%;
    color: #1f4e79;
    font-size: 22px;
    line-height: 1;
    transition: transform 0.2s ease, background 0.2s ease;
}
.section-toggle:hover::after { background: #eef3f8; }
.collapsible-card[open] > .section-toggle::after { transform: rotate(180deg); }
.section-content { padding: 0 28px 24px; }
.section-content > .note:first-child { margin-top: 0; }
.section-toggle:focus-visible,
.nav-tab:focus-visible,
.scene-person-button:focus-visible,
.scene-card:focus-visible,
.track-card:focus-visible,
.result.jump:focus-visible,
.transcript-line:focus-visible {
    outline: 3px solid #60a5fa;
    outline-offset: 3px;
}
.upload-box { text-align: center; padding: 36px; }
.upload-box h2 { margin-top: 0; margin-bottom: 8px; font-size: 24px; color: #1e293b; }
.upload-subtitle {
    color: #64748b; font-size: 15px; line-height: 1.6;
    margin: 0 auto 25px; max-width: 720px;
}
.file-input-wrapper {
    padding: 22px; border: 2px dashed #cbd5e1; border-radius: 10px;
    background: #f8fafc; margin-bottom: 5px;
}
.get-caption-btn {
    margin-top: 15px; padding: 13px 34px; border: none; border-radius: 8px;
    cursor: pointer; font-size: 16px; font-weight: 600;
    background: #1f4e79; color: white;
}
.get-caption-btn:hover { background: #163a5c; }
.note { color: #64748b; font-size: 14px; line-height: 1.6; }
video.player {
    width: 100%; max-height: 560px; background: #0f172a; border-radius: 10px;
}
.meta-row { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 14px; color: #334155; font-size: 14px; }
.timeline {
    position: relative; height: 42px; background: #e2e8f0; border-radius: 8px;
    overflow: hidden; cursor: pointer;
}
.timeline-scene {
    position: absolute; top: 0; height: 100%;
    border-right: 1px solid rgba(255,255,255,0.55);
}
.timeline-scene:hover { filter: brightness(1.08); }
.timeline-marker {
    position: absolute; top: 0; width: 2px; height: 100%; background: #0f172a;
    pointer-events: none;
}
.search-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.search-row input[type=text] {
    flex: 1; min-width: 220px; padding: 12px 14px; border: 1px solid #cbd5e1;
    border-radius: 8px; font-size: 15px;
}
.search-row select, .search-row button {
    padding: 12px 14px; border-radius: 8px; border: 1px solid #cbd5e1; font-size: 14px;
}
.search-row button {
    background: #1f4e79; color: white; border: none; font-weight: 600; cursor: pointer;
}
.search-row button:disabled { opacity: 0.65; cursor: wait; }
.transcript-line .orig-text { white-space: pre-wrap; }
.result {
    border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;
    margin-top: 10px; cursor: pointer; background: #f8fafc;
}
.result:hover { border-color: #9fb6cf; background: white; }
.result strong { color: #1f4e79; }
.chip {
    display: inline-block; background: #eef3f8; border-radius: 999px;
    padding: 4px 10px; margin: 3px 4px 3px 0; font-size: 13px; color: #334155;
}
.chip.action { background: #ecfdf5; color: #065f46; }
.chip.scene { background: #eff6ff; color: #1e40af; }
.scene-grid, .track-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 16px;
    align-items: start;
}
.scene-card, .track-card {
    border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; cursor: pointer;
    background: #fff; box-shadow: 0 2px 8px rgba(15, 23, 42, .04);
    transition: border-color .18s ease, box-shadow .18s ease;
}
.scene-card { position: relative; overflow: visible; }
.scene-card:hover, .track-card:hover {
    border-color: #b7c9dc; box-shadow: 0 8px 20px rgba(15, 23, 42, .09);
}
.scene-card:has(.scene-character-details[open]) { z-index: 5; }
.scene-card > img {
    width: 100%; height: 176px; object-fit: cover; display: block;
    border-radius: 11px 11px 0 0;
}
.scene-card .body, .track-card .body { padding: 16px; font-size: 14px; line-height: 1.55; }
.scene-card .body > strong { color: #163a5c; font-size: 14px; }
.scene-card .body > strong,
.track-card .body > strong { display: inline-block; margin-bottom: 5px; }
.scene-person-button {
    position: absolute; top: 10px; right: 10px; z-index: 1;
    display: inline-flex; align-items: center; justify-content: center; gap: 6px;
    min-width: 38px; height: 38px; padding: 0 10px; border: 1px solid #dbe4ee;
    border-radius: 999px; background: rgba(255,255,255,.96); color: #1f4e79;
    box-shadow: 0 2px 8px rgba(15,23,42,.16); cursor: pointer; font-weight: 700;
}
.scene-person-button:hover, .scene-person-button[aria-expanded="true"] { background: #1f4e79; color: white; }
.scene-person-button svg { width: 19px; height: 19px; fill: currentColor; }
.scene-characters, .video-characters-panel {
    padding: 14px; border: 1px solid #dbe5ef; border-radius: 11px;
    background: white; box-shadow: 0 12px 30px rgba(15, 23, 42, .16);
}
.scene-characters[hidden], .video-characters-panel[hidden] { display: none; }
.scene-character-details[open] .scene-characters {
    position: absolute; top: 54px; left: 50%; z-index: 5;
    width: min(440px, calc(100vw - 32px));
    max-height: min(78vh, 640px);
    transform: translateX(-50%);
    overflow-y: auto; overscroll-behavior: contain;
    padding: 18px;
}
.scene-characters > strong { display: block; margin-bottom: 8px; color: #163a5c; font-size: 17px; }
.scene-character-status { margin: 0 0 12px; font-size: 14px; }
.video-characters-toggle { margin: 0 0 12px; }
.video-characters-details > summary {
    list-style: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}
.video-characters-details > summary::after {
    content: "⌄";
    flex: 0 0 auto;
    width: 30px;
    height: 30px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 50%;
    font-size: 21px;
    line-height: 1;
    transition: transform 0.2s ease;
}
.video-characters-details[open] > summary::after { transform: rotate(180deg); }
.video-characters-details > summary::-webkit-details-marker,
.scene-character-details > summary::-webkit-details-marker { display: none; }
.scene-character-details > summary { list-style: none; }
.video-characters-panel { margin-top: 12px; }
.character-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 10px;
}
.scene-character-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
}
.video-character-grid {
    grid-template-columns: repeat(auto-fill, minmax(135px, 1fr));
    gap: 12px;
}
.character-profile {
    display: flex; align-items: flex-start; gap: 11px; min-width: 0;
    padding: 10px; border: 1px solid #e2e8f0; border-radius: 9px; background: #f8fafc;
}
.video-character-profile {
    flex-direction: column; align-items: center; text-align: center;
}
.character-photo {
    width: 76px; height: 76px; flex: 0 0 76px; object-fit: cover;
    border-radius: 9px; background: #e2e8f0;
}
.scene-character-grid .character-profile {
    align-items: center; gap: 10px; padding: 10px; border-radius: 10px;
}
.scene-character-grid .character-photo {
    width: 68px; height: 84px; flex-basis: 68px;
}
.scene-character-grid .character-info strong { font-size: 14px; }
.scene-character-grid .character-info { overflow-wrap: anywhere; }
.video-character-grid .character-profile {
    height: 100%; padding: 14px 10px; border-color: #e4eaf1;
    background: linear-gradient(180deg, #fff, #f8fafc);
}
.video-character-profile .character-photo {
    width: 128px; height: 128px; max-width: 100%; flex: 0 0 auto;
    border-radius: 12px;
}
.character-info { min-width: 0; }
.character-info strong { display: block; color: #1f4e79; margin-bottom: 4px; }
.character-info p { margin: 3px 0; overflow-wrap: anywhere; }
.video-character-profile .character-info { width: 100%; }
.video-character-profile .character-scenes { color: #475569; }
.character-empty { margin: 0; }
.warn { color: #b45309; background: #fffbeb; border: 1px solid #fde68a; padding: 10px 12px; border-radius: 8px; }
.info { color: #1e3a5f; background: #eff6ff; border: 1px solid #bfdbfe; padding: 10px 12px; border-radius: 8px; }
.library a { display: block; min-width: 0; text-decoration: none; color: inherit; }
.library .result strong {
    display: -webkit-box;
    overflow: hidden;
    overflow-wrap: anywhere;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    line-height: 1.4;
}
.library .result .note { overflow-wrap: anywhere; }
.library-toggle {
    list-style: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}
.library-toggle::-webkit-details-marker { display: none; }
.library-toggle::after {
    content: "⌃";
    flex: 0 0 auto;
    width: 30px;
    height: 30px;
    border: 1px solid #dbe5ef;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #1f4e79;
    font-size: 18px;
    line-height: 1;
    transform: translateY(3px);
    transition: transform 0.2s ease, background 0.2s ease;
}
.library-details:not([open]) .library-toggle::after {
    transform: rotate(180deg) translateY(-3px);
}
.library-toggle:hover::after { background: #eef3f8; }
.library-details { padding: 0 !important; }
.library-details > summary {
    padding: 24px 28px 18px;
}
.library-details > .library-body {
    padding: 0 28px 24px;
}
.transcript-line { padding: 8px 0; border-bottom: 1px solid #eef2f6; cursor: pointer; font-size: 14px; }
.transcript-line.active {
    background: #eff6ff;
    border-left: 3px solid #1f4e79;
    padding-left: 9px;
}
.speaker-label {
    display: inline-block;
    margin: 0 3px;
    padding: 2px 7px;
    border-radius: 999px;
    background: #eff6ff;
    color: #1e40af;
    font-weight: 700;
}
.transcript-line:hover { color: #1f4e79; }
h2 { color: #1e293b; margin-top: 0; }
@media (max-width: 700px) {
    body { padding: 20px 12px; }
    .header h1 { font-size: 28px; }
    .card { padding: 20px; }
    .section-toggle { padding: 18px 20px; }
    .section-content { padding: 0 20px 20px; }
    .scene-grid, .track-grid { grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); }
    .scene-card > img { height: 190px; }
    .scene-character-details[open] .scene-characters {
        left: 50%; right: auto; width: min(440px, calc(100vw - 32px));
        max-height: min(78vh, 640px);
        transform: translateX(-50%);
    }
    .scene-character-grid .character-photo { width: 72px; height: 90px; flex-basis: 72px; }
    .scene-character-grid .character-profile { padding: 12px; }
}
@media (max-width: 480px) {
    .scene-grid, .track-grid { grid-template-columns: 1fr; }
}
.progress-track { height: 10px; background: #e2e8f0; border-radius: 999px; overflow: hidden; margin-top: 8px; }
.progress-bar { height: 100%; background: #1f4e79; width: 0%; transition: width .3s; }
.progress-card { margin-top: 8px; }
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>VisualRAG</h1>
<p>Video Understanding &amp; Search</p>
</div>
<div class="nav-tabs">
<a class="nav-tab" href="/">Image Matching</a>
<a class="nav-tab active" href="/video">Video Understanding</a>
</div>

<div class="card upload-box">
<h2>Upload a video</h2>
<p class="upload-subtitle">
The pipeline samples frames, detects scenes, tracks people and objects,
recognizes actions, runs visual-evidence scene understanding, transcribes
speech, and builds a searchable timeline. Long videos keep working in the
background so this page can show results instead of going blank.
</p>
<form method="POST" action="/video" enctype="multipart/form-data">
<div class="file-input-wrapper">
<input type="file" name="video" accept="video/*" required>
</div>
<button type="submit" class="get-caption-btn">Analyze video</button>
</form>
</div>

{% if error %}
<div class="card"><p class="warn">{{ error }}</p></div>
{% endif %}

{% if library %}
<details class="card library-details" open>
<summary class="library-toggle">
<div>
<h2 style="margin-bottom:4px;">Analyzed videos</h2>
<p class="note" style="margin-bottom:0;">Open a finished analysis. The latest one is shown below. Re-upload the same file after this update to rebuild scenes, tracks, and actions.</p>
</div>
</summary>
<div class="library-body">
<div class="library">
{% for item in library %}
<a class="result" href="/video?id={{ item.video_id }}">
<strong title="{{ item.original_name }}">{{ item.original_name }}</strong>
<div class="note">{{ item.scenes }} scenes · {{ format_time_minutes(item.duration) }} · {{ item.sampled_frames }} frames
{% if item.has_transcript %}· {{ item.speech_status or "Completed" }}{% elif item.asr_error %}· {{ item.asr_error }}{% else %}· {{ item.speech_status or "No Speech Detected" }}{% endif %}
</div>
</a>
{% endfor %}
</div>
</details>
{% endif %}

{% if index or uploaded_filename %}
<details class="card collapsible-card" open>
<summary class="section-toggle"><h2>Player</h2></summary>
<div class="section-content">
<video id="player" class="player" src="/uploaded-video/{{ index.filename if index else uploaded_filename }}" controls preload="metadata"></video>
<p id="player-feedback" class="note" role="status" aria-live="polite" hidden></p>
{% if index %}
<div class="meta-row">
<span>Duration {{ format_time_minutes(index.duration) }}</span>
<span>{{ index.width }}×{{ index.height }}</span>
<span>{{ index.scenes|length }} scenes</span>
<span>{{ index.tracks|length }} tracks</span>
<span>{{ index.actions|length }} actions</span>
<span>{{ index.sampled_frames }} sampled frames</span>
</div>
{% elif job_id %}
<div class="progress-card">
  <p class="note" id="job-status" role="status" aria-live="polite">Upload received. Preparing video analysis…</p>
  <div class="progress-track"><div class="progress-bar" id="job-progress" style="width:0%"></div></div>
  <button type="button" class="get-caption-btn" id="stop-job" style="margin-top:12px;">Stop analysis</button>
</div>
{% endif %}
</div>
</details>
{% endif %}

{% if index %}
<details class="card collapsible-card">
<summary class="section-toggle"><h2>Video timeline / index</h2></summary>
<div class="section-content">
<p class="note">Each block is a detected scene. Click to jump. The dark marker follows playback.</p>
<div class="timeline" id="timeline">
{% for scene in index.scenes %}
<div class="timeline-scene"
     title="Scene {{ scene.id }} {{ format_time_minutes(scene.start) }}–{{ format_time_minutes(scene.end) }}"
     data-start="{{ scene.start }}"
     style="left: {{ (scene.start / index.duration * 100) if index.duration else 0 }}%;
            width: {{ (scene.duration / index.duration * 100) if index.duration else 0 }}%;
            background: hsl({{ (scene.id * 47) % 360 }}, 42%, 62%);">
</div>
{% endfor %}
<div class="timeline-marker" id="timeline-marker"></div>
</div>
</div>
</details>

<details class="card collapsible-card">
<summary class="section-toggle"><h2>Search the video</h2></summary>
<div class="section-content">
<p class="note">
Word/phrase frequency (speech), visual concept search, semantic questions
such as “When does the person start talking on the phone?”, or multimodal
(visual + speech + time). Click a result to jump.
</p>
<div class="search-row">
<input id="query" type="text" placeholder="phone, person, When does the person start talking on the phone?">
<select id="mode">
<option value="multimodal">Multimodal</option>
<option value="phrase">Word / phrase (speech)</option>
<option value="visual">Visual concept</option>
<option value="semantic">Semantic</option>
</select>
<button type="button" id="search-btn">Search</button>
</div>
<p id="search-summary" class="note"></p>
<div id="results"></div>
</div>
</details>

<details class="card collapsible-card">
<summary class="section-toggle"><h2>Scenes</h2></summary>
<div class="section-content">
<p class="note">Select a scene, object track, action, or transcript entry to jump the video to its start.</p>
{% set person_tracks = index.tracks|selectattr("label", "equalto", "person")|list %}
<details class="video-characters-details">
<summary class="get-caption-btn video-characters-toggle">Characters in this video</summary>
<div class="video-characters-panel">
<p class="note">Names are shown only when the video provides clear evidence linking one name to one character.</p>
{% if person_tracks %}
<p class="note character-empty" id="video-character-status">Open this panel to load character matches.</p>
<div class="character-grid video-character-grid" id="video-character-grid"></div>
{% else %}
<p class="note character-empty">No person tracks were detected in this video.</p>
{% endif %}
</div>
</details>
<div class="scene-grid">
{% for scene in index.scenes %}
<div class="scene-card" data-start="{{ scene.start }}" tabindex="0" role="group"
     aria-label="Play scene {{ scene.id }} from {{ format_time_minutes(scene.start) }}">
<details class="scene-character-details">
<summary class="scene-person-button" aria-label="Show characters in scene {{ scene.id }}"
        title="Characters in this scene">
<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm0 2c-4.42 0-8 2.24-8 5v1h16v-1c0-2.76-3.58-5-8-5Z"/></svg>
</summary>
<div class="scene-characters">
<strong>Characters in Scene {{ scene.id }}</strong>
<p class="note scene-character-status">Open to load characters for this scene.</p>
<div class="character-grid scene-character-grid" data-scene-id="{{ scene.id }}" data-frame="{{ scene.keyframe }}"></div>
</div>
</details>
<img src="/video-frame/{{ scene.keyframe }}" alt="Scene {{ scene.id }}">
<div class="body">
<strong>Scene {{ scene.id }}</strong>
<div>{{ format_time_minutes(scene.start) }} – {{ format_time_minutes(scene.end) }}
<div>{{ format_time_minutes(scene.duration) }}</div>
</div>
{% if scene.contents and scene.contents.Scene %}
{% for item in scene.contents.Scene[:1] %}
<span class="chip scene">{{ item }}</span>
{% endfor %}
{% endif %}
{% if scene.contents and scene.contents.People %}
{% for item in scene.contents.People[:3] %}
<span class="chip">{{ item }}</span>
{% endfor %}
{% endif %}
{% for obj in scene.objects %}
<span class="chip">{{ obj }}</span>
{% endfor %}
{% for action in scene.actions[:4] %}
<span class="chip action">{{ action }}</span>
{% endfor %}
{% if scene.transcript %}
<p class="note">{{ scene.transcript[:140] }}</p>
{% endif %}
</div>
</div>
{% endfor %}
</div>
</div>
</details>

<details class="card collapsible-card">
<summary class="section-toggle"><h2>Object tracks</h2></summary>
<div class="section-content">
<p class="note">Two detectors with standard COCO object labels are combined to improve coverage. Small, obscured, unusual, or out-of-category objects can still be missed or mislabeled; detections are estimates, not guaranteed ground truth.</p>
<div class="track-grid">
{% for track in index.tracks %}
<div class="track-card" data-start="{{ track.start }}" tabindex="0" role="button"
     aria-label="Play {{ track.label }} track from {{ format_time_minutes(track.start) }}">
<div class="body">
<strong>{{ track.label }}</strong> · track {{ track.track_id }}
<div>{{ format_time_minutes(track.start) }} – {{ format_time_minutes(track.end) }}
<div>{{ format_time_minutes(track.duration) }}</div>
</div>
<div class="note">{{ track.hits }} hits · {{ "%.0f"|format(track.confidence * 100) }}% conf
{% if track.scene_ids %} · scenes {{ track.scene_ids|join(', ') }}{% endif %}
</div>
</div>
</div>
{% endfor %}
</div>
{% if not index.tracks %}
<p class="note">No persistent tracks were found in the sampled frames.</p>
{% endif %}
</div>
</details>

<details class="card collapsible-card">
<summary class="section-toggle"><h2>Detected actions</h2></summary>
<div class="section-content">
{% for action in index.actions %}
<div class="result jump" data-start="{{ action.start }}" tabindex="0" role="button"
     aria-label="Play action from {{ format_time_minutes(action.start) }}">
<strong>{{ format_time_minutes(action.start) }} – {{ format_time_minutes(action.end) }}</strong>
<div>{{ action.label }}{% if action.scene_id %} · scene {{ action.scene_id }}{% endif %}</div>
</div>
{% endfor %}
{% if not index.actions %}
<p class="note">No timed actions were detected.</p>
{% endif %}
</div>
</details>

<details class="card collapsible-card">
<summary class="section-toggle"><h2>Speech transcript</h2></summary>
<div class="section-content">
<div class="search-row" style="margin-bottom:8px;">
  <label for="translation-language">Translate transcript to:</label>
  <select id="translation-language">
    <option value="">Original transcript</option>
    <option value="en">English</option>
    <option value="hi">Hindi</option>
    <option value="te">Telugu</option>
    <option value="ta">Tamil</option>
    <option value="es">Spanish</option>
    <option value="fr">French</option>
    <option value="de">German</option>
    <option value="ja">Japanese</option>
    <option value="zh">Chinese</option>
    <option value="ko">Korean</option>
    <option value="ar">Arabic</option>
    <option value="ru">Russian</option>
    <option value="pt">Portuguese</option>
    <option value="it">Italian</option>
    <option value="nl">Dutch</option>
    <option value="pl">Polish</option>
    <option value="tr">Turkish</option>
    <option value="vi">Vietnamese</option>
    <option value="th">Thai</option>
    <option value="id">Indonesian</option>
    <option value="ms">Malay</option>
    <option value="fil">Filipino</option>
    <option value="uk">Ukrainian</option>
    <option value="cs">Czech</option>
    <option value="ro">Romanian</option>
    <option value="el">Greek</option>
    <option value="he">Hebrew</option>
    <option value="fa">Persian</option>
    <option value="ur">Urdu</option>
    <option value="bn">Bengali</option>
    <option value="gu">Gujarati</option>
    <option value="kn">Kannada</option>
    <option value="ml">Malayalam</option>
    <option value="mr">Marathi</option>
    <option value="pa">Punjabi</option>
  </select>
</div>
<div id="translate-status" class="note" style="margin-bottom:8px;">Choose a language to translate the transcript automatically.</div>
{% if index.asr_error %}
<p class="warn" id="asr-error">Speech transcription failed: {{ index.asr_error }}</p>
<button type="button" class="get-caption-btn" id="retry-asr">Retry speech transcription</button>
{% endif %}
<div id="transcript-container">
{% for seg in index.transcript %}
<div class="transcript-line jump" data-start="{{ seg.start }}" data-end="{{ seg.end }}"
     tabindex="0" role="button" aria-label="Play transcript from {{ format_time_minutes(seg.start) }}">
<strong>{{ seg.start|format_time_minutes }} - {{ seg.end|format_time_minutes }}</strong>{% if seg.speaker %} <span class="speaker-label">{{ seg.speaker }}</span>{% endif %} — <span class="orig-text" data-original="{{ seg.text }}">{{ seg.text }}</span>
</div>
{% endfor %}
</div>
{% if not index.transcript and not index.asr_error %}
<p class="note">No speech was detected.</p>
{% endif %}
</div>
</details>

{% endif %}
{% if index %}
<details class="card collapsible-card">
<summary class="section-toggle"><h2>Audio &amp; Sound Events</h2></summary>
<div class="section-content">
{% if index.audio_error %}
<p class="warn">Sound-event analysis unavailable: {{ index.audio_error }}</p>
{% else %}
<div class="note" style="margin-bottom:12px;">Classified with a pretrained AudioSet multi-label model in overlapping 5-second audio windows. Event times are approximate windows; labels and confidence scores are model estimates, not guaranteed ground truth.</div>
{% for ev in index.audio_events %}
<div class="result jump" data-start="{{ ev.start }}" tabindex="0" role="button"
     aria-label="Play sound event from {{ ev.start|format_time_minutes }}">
<strong>{{ ev.start|format_time_minutes }} - {{ ev.end|format_time_minutes }}</strong>
<div>{{ ev.label }} - {{ "%.0f"|format(ev.confidence * 100) }}% model score</div>
</div>
{% endfor %}
{% if not index.audio_events %}
<p class="note">No sound events passed the model confidence threshold. Speech transcription, when available, is shown separately above.</p>
{% endif %}
{% endif %}
</div>
</details>
{% endif %}

</div>

<script>
(function () {
  const player = document.getElementById("player");
  const playerFeedback = document.getElementById("player-feedback");
  const marker = document.getElementById("timeline-marker");
  const timeline = document.getElementById("timeline");
  const videoId = {{ (index.video_id if index else video_id) | tojson }};
  const jobId = {{ (job_id if job_id else "")|tojson }};

  let characterGroups = null;
  let characterGroupsRequest = null;

  function loadCharacterGroups() {
    if (characterGroups) return Promise.resolve(characterGroups);
    if (characterGroupsRequest) return characterGroupsRequest;
    characterGroupsRequest = fetch("/video/characters/" + encodeURIComponent(videoId), {
      cache: "no-store"
    }).then(function (response) {
      if (!response.ok) throw new Error("Character loading failed (" + response.status + ").");
      return response.json();
    }).then(function (data) {
      characterGroups = data.characters || [];
      return characterGroups;
    }).catch(function (error) {
      characterGroupsRequest = null;
      throw error;
    });
    return characterGroupsRequest;
  }

  function createCharacterCard(character, index, sceneId, fallbackFrame) {
    const card = document.createElement("div");
    card.className = "character-profile" + (sceneId ? "" : " video-character-profile");
    const image = document.createElement("img");
    image.className = "character-photo";
    image.alt = "Character " + index;
    const sceneTrackId = sceneId && character.scene_tracks[String(sceneId)];
    if (sceneTrackId) {
      image.src = "/video-character/" + encodeURIComponent(videoId) + "/" +
        encodeURIComponent(sceneId) + "/" + encodeURIComponent(sceneTrackId);
    } else if (fallbackFrame || character.portrait_frame) {
      image.src = "/video-frame/" + encodeURIComponent(fallbackFrame || character.portrait_frame);
    }
    card.appendChild(image);
    const info = document.createElement("div");
    info.className = "character-info";
    if (character.character_name) {
      const name = document.createElement("strong");
      name.textContent = character.character_name;
      info.appendChild(name);
    }
    if (!sceneId) {
      const scenes = document.createElement("p");
      scenes.className = "note character-scenes";
      scenes.textContent = "Scenes: " + character.scene_ids.join(", ");
      info.appendChild(scenes);
    }
    card.appendChild(info);
    return card;
  }

  function renderVideoCharacters(groups) {
    const grid = document.getElementById("video-character-grid");
    const status = document.getElementById("video-character-status");
    if (!grid || !status) return;
    grid.replaceChildren();
    if (!groups.length) {
      status.textContent = "No person tracks were detected in this video.";
      return;
    }
    groups.forEach(function (character, index) {
      grid.appendChild(createCharacterCard(character, index + 1, null));
    });
    status.remove();
  }

  function renderSceneCharacters(container, groups) {
    const sceneId = Number(container.dataset.sceneId);
    const sceneGroups = groups.filter(function (character) {
      return character.scene_ids.indexOf(sceneId) !== -1;
    });
    const status = container.previousElementSibling;
    container.replaceChildren();
    if (!sceneGroups.length) {
      status.textContent = "No person tracks were detected in this scene.";
      return;
    }
    sceneGroups.forEach(function (character, index) {
      container.appendChild(createCharacterCard(character, index + 1, sceneId, container.dataset.frame));
    });
    status.textContent = "Only confidently matched character names are shown.";
  }

  document.addEventListener("toggle", function (event) {
    const details = event.target;
    if (!(details instanceof HTMLDetailsElement) || !details.open || !videoId) return;
    if (details.matches(".video-characters-details")) {
      const status = document.getElementById("video-character-status");
      if (characterGroups) {
        renderVideoCharacters(characterGroups);
        return;
      }
      if (status) status.textContent = "Loading character matches…";
      loadCharacterGroups().then(renderVideoCharacters).catch(function () {
        if (status) status.textContent = "Could not load character matches. Close and reopen this panel to retry.";
      });
    } else if (details.matches(".scene-character-details")) {
      const container = details.querySelector(".scene-character-grid");
      const status = details.querySelector(".scene-character-status");
      if (!container || container.dataset.loaded === "true") return;
      if (characterGroups) {
        renderSceneCharacters(container, characterGroups);
        container.dataset.loaded = "true";
        return;
      }
      if (status) status.textContent = "Loading characters…";
      loadCharacterGroups().then(function (groups) {
        renderSceneCharacters(container, groups);
        container.dataset.loaded = "true";
      }).catch(function () {
        if (status) status.textContent = "Could not load characters. Close and reopen to retry.";
      });
    }
  }, true);

  if (jobId) {
    const statusEl = document.getElementById("job-status");
    const stopButton = document.getElementById("stop-job");
    let pollDelay = 3000;
    let stopping = false;

    if (stopButton) {
      stopButton.addEventListener("click", function () {
        if (stopping) return;
        stopping = true;
        stopButton.disabled = true;
        stopButton.textContent = "Stopping…";
        if (statusEl) statusEl.textContent = "Stopping the queued analysis…";
        fetch("/video/job/" + jobId + "/cancel", {
          method: "POST",
          headers: { "Content-Type": "application/json" }
        }).then(function (r) { return r.json(); }).then(function (data) {
          if (statusEl) {
            statusEl.className = "note";
            statusEl.textContent = data.message || "Analysis stopped.";
          }
          stopButton.textContent = "Analysis stopped";
        }).catch(function () {
          stopping = false;
          stopButton.disabled = false;
          stopButton.textContent = "Stop analysis";
          if (statusEl) statusEl.textContent = "Could not stop the analysis. Please try again.";
        });
      });
    }

    function pollJob() {
      fetch("/video/job/" + jobId, { cache: "no-store" }).then(function (r) {
        if (!r.ok) throw new Error("Progress request failed (" + r.status + ").");
        return r.json();
      }).then(function (data) {
        pollDelay = 3000;
        if (statusEl) {
      statusEl.textContent = data.message || data.status || "Working…";
      const bar = document.getElementById("job-progress");
       if (bar && typeof data.progress === "number") bar.style.width = Math.max(0, Math.min(100, data.progress)) + "%";
       const step = String(data.message || "").match(/Step ([0-9]+)[/]5/);
      statusEl.dataset.step = step ? step[1] : "";
    }
        if (data.status === "done" && data.video_id) {
          window.history.replaceState({}, "", "/video?id=" + encodeURIComponent(data.video_id));
          window.location.reload();
          return;
        }
        if (data.status === "error") {
          if (statusEl) {
            statusEl.className = "warn";
            statusEl.textContent = data.message || "Analysis failed.";
          }
          if (stopButton) stopButton.style.display = "none";
          return;
        }
        if (data.status === "cancelled") {
          if (statusEl) {
            statusEl.className = "note";
            statusEl.textContent = data.message || "Analysis stopped.";
          }
          const bar = document.getElementById("job-progress");
          if (bar) bar.style.width = "0%";
          if (stopButton) {
            stopButton.disabled = true;
            stopButton.textContent = "Analysis stopped";
          }
          return;
        }
        setTimeout(pollJob, pollDelay);
      }).catch(function (err) {
        if (statusEl) statusEl.textContent = "Still working. " + err.message;
        pollDelay = Math.min(12000, pollDelay * 1.5);
        setTimeout(pollJob, pollDelay);
      });
    }
    pollJob();
  }

  const retryAsr = document.getElementById("retry-asr");
  if (retryAsr && videoId) {
    retryAsr.addEventListener("click", function () {
      retryAsr.disabled = true;
      retryAsr.textContent = "Starting transcription…";
      fetch("/video/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: videoId, language: "" })
      }).then(function (r) { return r.json(); }).then(function (data) {
        if (data.error) {
          retryAsr.disabled = false;
          retryAsr.textContent = "Retry speech transcription";
          alert(data.error);
          return;
        }
        window.location.href = "/video?id=" + videoId + "&job=" + data.job_id;
      }).catch(function (err) {
        retryAsr.disabled = false;
        retryAsr.textContent = "Retry speech transcription";
        alert(err);
      });
    });
  }

  function jump(seconds) {
    if (!player) return;
    const target = Math.max(0, Number(seconds) || 0);

    const scrollToPlayer = function () {
      const playerCard = player.closest(".card");
      if (!playerCard) return;
      const cardTop = playerCard.getBoundingClientRect().top + window.scrollY - 16;
      window.scrollTo({ top: Math.max(0, cardTop), behavior: "auto" });
    };

    const startPlayback = function () {
      const duration = Number.isFinite(player.duration) ? player.duration : target;
      player.currentTime = Math.min(target, Math.max(0, duration));
      if (playerFeedback) playerFeedback.hidden = true;
      const playResult = player.play();
      if (playResult && typeof playResult.catch === "function") {
        playResult.catch(function () {
          if (playerFeedback) {
            playerFeedback.textContent = "Playback could not start. Use the video controls to retry.";
            playerFeedback.hidden = false;
          }
        });
      }
      scrollToPlayer();
    };

    if (player.readyState < HTMLMediaElement.HAVE_METADATA) {
      player.addEventListener("loadedmetadata", startPlayback, { once: true });
    } else {
      startPlayback();
    }
  }

  document.querySelectorAll(
    ".scene-card[data-start], .track-card[data-start], .result.jump[data-start], .transcript-line[data-start]"
  ).forEach(function (el) {
    el.addEventListener("click", function (event) {
      if (event.target.closest(".scene-character-details, .video-characters-details, summary")) return;
      event.stopPropagation();
      if (el.classList.contains("transcript-line")) {
        document.querySelectorAll(".transcript-line.active").forEach(function (line) {
          line.classList.remove("active");
        });
        el.classList.add("active");
      }
      jump(el.getAttribute("data-start"));
    });
    el.addEventListener("keydown", function (event) {
      if (event.key !== "Enter" && event.key !== " ") return;
      if (event.target.closest(".scene-character-details, .video-characters-details, summary")) return;
      event.preventDefault();
      if (el.classList.contains("transcript-line")) {
        document.querySelectorAll(".transcript-line.active").forEach(function (line) {
          line.classList.remove("active");
        });
        el.classList.add("active");
      }
      jump(el.getAttribute("data-start"));
    });
  });

  if (timeline && player) {
    timeline.addEventListener("click", function (event) {
      const rect = timeline.getBoundingClientRect();
      const ratio = (event.clientX - rect.left) / rect.width;
      jump(ratio * (player.duration || 0));
    });
    player.addEventListener("timeupdate", function () {
      if (!marker || !player.duration) return;
      marker.style.left = (player.currentTime / player.duration * 100) + "%";
    });
  }

  if (player) {
    player.addEventListener("timeupdate", function () {
      const lines = transcriptLines();
      if (!lines.length) return;
      let active = null;
      for (let i = 0; i < lines.length; i++) {
        const start = Number(lines[i].getAttribute("data-start")) || 0;
        const end = Number(lines[i].getAttribute("data-end")) || Number.MAX_VALUE;
        if (player.currentTime >= start && player.currentTime < end) {
          active = lines[i];
          break;
        }
      }
      lines.forEach(function (line) {
        line.classList.toggle("active", line === active);
      });
    });
  }

  function formatTimeMinutes(seconds) {
    if (seconds === null || seconds === undefined || isNaN(seconds)) {
      return "00:00.00";
    }
    const totalCentiseconds = Math.max(0, Math.round(Number(seconds) * 100));
    const hours = Math.floor(totalCentiseconds / 360000);
    const minutes = Math.floor((totalCentiseconds % 360000) / 6000);
    const secs = Math.floor((totalCentiseconds % 6000) / 100);
    const hundredths = totalCentiseconds % 100;
    const ss = String(secs).padStart(2, "0");
    const cc = String(hundredths).padStart(2, "0");
    if (hours > 0) {
      return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${ss}.${cc}`;
    }
    return `${String(minutes).padStart(2, "0")}:${ss}.${cc}`;
  }

  const searchBtn = document.getElementById("search-btn");
  if (searchBtn) {
    searchBtn.addEventListener("click", runSearch);
    document.getElementById("query").addEventListener("keydown", function (e) {
      if (e.key === "Enter") runSearch();
    });
  }

  function runSearch() {
    const query = document.getElementById("query").value;
    const mode = document.getElementById("mode").value;
    const summary = document.getElementById("search-summary");
    const box = document.getElementById("results");
    box.innerHTML = "<p class='note'>Searching…</p>";
    fetch("/video/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId, query: query, mode: mode })
    }).then(function (r) {
      return r.json().catch(function () {
        throw new Error("Server returned an invalid response (" + r.status + ").");
      });
    }).then(function (data) {
      if (data.error) {
        summary.textContent = data.error;
        box.innerHTML = "";
        return;
      }
      let extra = "";
      if (data.phrase_count !== undefined) {
        extra = " Speech hits: " + data.phrase_count + ". Visual hits: " + data.visual_count + ".";
      } else if (mode === "phrase" || mode === "word") {
        extra = " " + data.count + " occurrence" + (data.count === 1 ? "" : "s") + " found.";
      }
      if (mode === "phrase" || mode === "word") {
        const occurrenceLabel = data.count === 1 ? "1 occurrence" : data.count + " occurrences";
        summary.textContent = occurrenceLabel + " found for: " + String(data.query || "") + ". Click a result to jump.";
      } else {
        summary.textContent = data.count + " result(s) for: " + String(data.query || "") + "." + extra;
      }
      box.innerHTML = "";
      (data.results || []).forEach(function (item, index) {
        const div = document.createElement("div");
        div.className = "result";
        const time = document.createElement("span");
        time.className = "time";
        time.textContent = formatTimeMinutes(item.start) + (item.end ? " – " + formatTimeMinutes(item.end) : "");
        const heading = document.createElement("strong");
        heading.textContent = (mode === "phrase" || mode === "word")
          ? "Occurrence " + (item.occurrence_number || (index + 1)) + " · " + item.label
          : item.label;
        const title = document.createElement("span");
        title.textContent = " · " + (item.title || "");
        const evidence = document.createElement("div");
        evidence.className = "note";
        evidence.textContent = item.evidence || "";
        div.append(time, heading, title, evidence);
        div.addEventListener("click", function () { jump(item.start); });
        box.appendChild(div);
      });
      if (!data.results || !data.results.length) {
        box.innerHTML = "<p class='note'>No matches in the video index.</p>";
      }
    }).catch(function (err) {
      summary.textContent = "Search failed.";
      box.innerHTML = "<p class='warn'>" + err + "</p>";
    });
  }
})();

function transcriptLines() {
  const container = document.getElementById("transcript-container");
  return container ? Array.from(container.querySelectorAll(".transcript-line")) : [];
}

function setTranscriptLineText(el, text) {
  if (!el) return;
  let orig = el.querySelector(".orig-text");
  if (!orig) {
    orig = document.createElement("span");
    orig.className = "orig-text";
    el.appendChild(orig);
  }
  if (!orig.getAttribute("data-original")) orig.setAttribute("data-original", orig.textContent || "");
  orig.textContent = text;
}

function restoreOriginalTranscript() {
  transcriptLines().forEach(function (el) {
    const orig = el.querySelector(".orig-text");
    if (!orig) return;
    orig.textContent = orig.getAttribute("data-original") || orig.textContent;
  });
}

function applyTranslatedSegments(segs) {
  const lines = transcriptLines();
  segs.forEach(function (seg, idx) {
    let line = lines[idx];
    if (seg && seg.start != null) {
      const match = lines.find(function (el) {
        return Math.abs(Number(el.getAttribute("data-start")) - Number(seg.start)) < 0.08;
      });
      if (match) line = match;
    }
    setTranscriptLineText(line, seg.translated_text || seg.text || "");
  });
}

function translateViaBrowser(texts, lang) {
  const delimiter = "\\n⟦S⟧\\n";
  const browserLanguageCodes = {
    zh: "zh-CN",
    fil: "tl",
    he: "iw",
  };
  const browserLang = browserLanguageCodes[lang] || lang;
  const chunks = [];
  let current = [];
  let size = 0;
  texts.forEach(function (text, idx) {
    const extra = String(text || "").length + delimiter.length;
    if (current.length && size + extra > 3000) {
      chunks.push(current);
      current = [];
      size = 0;
    }
    current.push({ idx: idx, text: String(text || "") });
    size += extra;
  });
  if (current.length) chunks.push(current);

  function translateChunk(chunk) {
    const joined = chunk.map(function (item) { return item.text; }).join(delimiter);
    const url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl="
      + encodeURIComponent(browserLang) + "&dt=t&q=" + encodeURIComponent(joined);
    const controller = new AbortController();
    const timeout = window.setTimeout(function () { controller.abort(); }, 15000);
    function request(query) {
      const requestUrl = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl="
        + encodeURIComponent(browserLang) + "&dt=t&q=" + encodeURIComponent(query);
      return fetch(requestUrl, { signal: controller.signal }).then(function (r) {
      if (!r.ok) throw new Error("Browser translation failed (" + r.status + ")");
      return r.json();
      });
    }
    return request(joined).then(function (data) {
      const translated = ((data && data[0]) || []).map(function (part) { return part && part[0] ? part[0] : ""; }).join("");
      if (translated.trim() === joined.trim() && joined !== joined.toLowerCase()) {
        return request(joined.toLowerCase()).then(function (lowerData) {
          return (lowerData && lowerData[0] || []).map(function (part) { return part && part[0] ? part[0] : ""; }).join("");
        });
      }
      return translated;
    }).then(function (translated) {
      const parts = translated.split(/\\s*⟦S⟧\\s*/).map(function (part) {
        return part.trim();
      }).filter(function (part, index) {
        return part || chunk.length === 1;
      });
      if (parts.length === chunk.length) return parts;

      // Google Translate may omit or rewrite a batching marker. Translating
      // each line separately avoids losing the transcript alignment.
      return Promise.all(chunk.map(function (item) {
        return request(item.text).then(function (data) {
          const part = ((data && data[0]) || []).map(function (segment) {
            return segment && segment[0] ? segment[0] : "";
          }).join("");
          if (!item.text.trim() || part.trim() === item.text.trim()) return item.text;
          return part;
        });
      }));
    }).finally(function () { window.clearTimeout(timeout); });
  }

  return chunks.reduce(function (promise, chunk) {
    return promise.then(function (all) {
      return translateChunk(chunk).then(function (parts) {
        parts.forEach(function (part, i) { all[chunk[i].idx] = part; });
        return all;
      });
    });
  }, Promise.resolve(texts.slice()));
}

let transcriptTranslationRequestId = 0;

function translateTranscript() {
  const langSelect = document.getElementById("translation-language");
  const lang = langSelect ? langSelect.value : "";
  const requestId = ++transcriptTranslationRequestId;
  const statusEl = document.getElementById("translate-status");
  const videoId = {{ (index.video_id if index else video_id) | tojson }};
  const selectedLabel = langSelect && langSelect.selectedIndex >= 0
    ? langSelect.options[langSelect.selectedIndex].text
    : lang;

  if (!lang) {
    restoreOriginalTranscript();
    if (statusEl) {
      statusEl.className = "note";
      statusEl.textContent = "Showing original transcript.";
    }
    return false;
  }
  if (!transcriptLines().length) {
    if (statusEl) {
      statusEl.className = "warn";
      statusEl.textContent = "No transcript is available to translate.";
    }
    return false;
  }
  if (statusEl) {
    statusEl.className = "info";
    statusEl.textContent = "Translating transcript to " + selectedLabel + "...";
  }
  const originalTexts = transcriptLines().map(function (el) {
    const orig = el.querySelector(".orig-text");
    return (orig && (orig.getAttribute("data-original") || orig.textContent)) || "";
  });

  function finishOk(segs) {
    if (requestId !== transcriptTranslationRequestId) return;
    applyTranslatedSegments(segs);
    if (statusEl) {
      statusEl.className = "info";
      statusEl.textContent = "Transcript translated to " + selectedLabel + ".";
    }
  }

  function finishErr(err) {
    if (requestId !== transcriptTranslationRequestId) return;
    if (statusEl) {
      statusEl.className = "warn";
      statusEl.textContent = "Translation failed: " + (err && err.message ? err.message : err);
    }
  }

  const serverController = new AbortController();
  const serverTimeout = window.setTimeout(function () { serverController.abort(); }, 12000);
  const sourceLanguage = {{ ((index.source_language or index.transcription_language if index else "") or "") | tojson }};
  const serverCall = videoId ? fetch("/video/translate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_id: videoId, language: lang, source_language: sourceLanguage || null }),
    signal: serverController.signal
  }).then(function (r) {
    return r.json().then(function (data) {
      if (!r.ok || (data && data.error)) throw new Error((data && data.error) || ("Translation failed (" + r.status + ")"));
      return data;
    }).finally(function () { window.clearTimeout(serverTimeout); });
  }) : Promise.reject(new Error("No video ID"));

  serverCall.then(function (data) {
    const segs = data.segments || [];
    if (!segs.length) throw new Error("The translator returned no transcript lines.");
    finishOk(segs);
  }).catch(function () {
    if (requestId !== transcriptTranslationRequestId) return [];
    return translateViaBrowser(originalTexts, lang).then(function (translated) {
      finishOk(translated.map(function (text, idx) {
        const line = transcriptLines()[idx];
        return {
          start: line ? line.getAttribute("data-start") : idx,
          translated_text: text
        };
      }));
    });
  }).catch(finishErr);
  return false;
}

document.addEventListener("DOMContentLoaded", function () {
  const translationLanguage = document.getElementById("translation-language");
  if (translationLanguage) translationLanguage.addEventListener("change", translateTranscript);
});
</script>
</body>
</html>
"""
