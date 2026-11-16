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
.container { width: min(1480px, 100%); margin: 0 auto; }
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
.result {
    border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;
    margin-top: 10px; cursor: pointer; background: #f8fafc;
}
.result:hover { border-color: #1f4e79; background: white; }
.result strong { color: #1f4e79; }
.chip {
    display: inline-block; background: #eef3f8; border-radius: 999px;
    padding: 4px 10px; margin: 3px 4px 3px 0; font-size: 13px; color: #334155;
}
.chip.action { background: #ecfdf5; color: #065f46; }
.chip.scene { background: #eff6ff; color: #1e40af; }
.scene-grid, .track-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px;
}
.scene-card, .track-card {
    border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; cursor: pointer;
    background: #f8fafc;
}
.scene-card img { width: 100%; height: 140px; object-fit: cover; display: block; }
.scene-card .body, .track-card .body { padding: 10px 12px; font-size: 13px; }
.warn { color: #b45309; background: #fffbeb; border: 1px solid #fde68a; padding: 10px 12px; border-radius: 8px; }
.info { color: #1e3a5f; background: #eff6ff; border: 1px solid #bfdbfe; padding: 10px 12px; border-radius: 8px; }
.library a { display: block; text-decoration: none; color: inherit; }
.transcript-line { padding: 8px 0; border-bottom: 1px solid #eef2f6; cursor: pointer; font-size: 14px; }
.transcript-line:hover { color: #1f4e79; }
h2 { color: #1e293b; margin-top: 0; }
@media (max-width: 700px) {
    body { padding: 20px 12px; }
    .header h1 { font-size: 28px; }
    .card { padding: 20px; }
}
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

{% if job_id %}
<div class="card" id="job-card">
<h2>Processing video</h2>
<p class="info" id="job-status">Working — this can take several minutes for a long clip. Keep this tab open.</p>
</div>
{% endif %}

{% if library %}
<div class="card">
<h2>Analyzed videos</h2>
<p class="note">Open a finished analysis. The latest one is shown below. Re-upload the same file after this update to rebuild scenes, tracks, and actions.</p>
<div class="library">
{% for item in library %}
<a class="result" href="/video?id={{ item.video_id }}">
<strong>{{ item.original_name }}</strong>
<div class="note">{{ item.scenes }} scenes · {{ "%.0f"|format(item.duration) }}s · {{ item.sampled_frames }} frames
{% if item.has_transcript %}· transcript ready{% elif item.asr_error %}· speech pending{% endif %}
</div>
</a>
{% endfor %}
</div>
</div>
{% endif %}

{% if index or uploaded_filename %}
<div class="card">
<h2>Player</h2>
<video id="player" class="player" src="/uploaded-video/{{ index.filename if index else uploaded_filename }}" controls preload="metadata"></video>
{% if index %}
<div class="meta-row">
<span>Duration {{ "%.1f"|format(index.duration) }}s</span>
<span>{{ index.width }}×{{ index.height }}</span>
<span>{{ index.scenes|length }} scenes</span>
<span>{{ index.tracks|length }} tracks</span>
<span>{{ index.actions|length }} actions</span>
<span>{{ index.sampled_frames }} sampled frames</span>
</div>
{% elif job_id %}
<p class="note">Your uploaded video is ready to play. VisualRAG is analyzing it in the background; results will appear when processing finishes.</p>
{% endif %}
</div>
{% endif %}

{% if index %}
<div class="card">
<h2>Video timeline / index</h2>
<p class="note">Each block is a detected scene. Click to jump. The dark marker follows playback.</p>
<div class="timeline" id="timeline">
{% for scene in index.scenes %}
<div class="timeline-scene"
     title="Scene {{ scene.id }} {{ '%.1f'|format(scene.start) }}s–{{ '%.1f'|format(scene.end) }}s"
     data-start="{{ scene.start }}"
     style="left: {{ (scene.start / index.duration * 100) if index.duration else 0 }}%;
            width: {{ (scene.duration / index.duration * 100) if index.duration else 0 }}%;
            background: hsl({{ (scene.id * 47) % 360 }}, 42%, 62%);">
</div>
{% endfor %}
<div class="timeline-marker" id="timeline-marker"></div>
</div>
</div>

<div class="card">
<h2>Search the video</h2>
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

<div class="card">
<h2>Scenes</h2>
<div class="scene-grid">
{% for scene in index.scenes %}
<div class="scene-card" data-start="{{ scene.start }}">
<img src="/video-frame/{{ scene.keyframe }}" alt="Scene {{ scene.id }}">
<div class="body">
<strong>Scene {{ scene.id }}</strong>
<div>{{ "%.1f"|format(scene.start) }}s – {{ "%.1f"|format(scene.end) }}s
({{ "%.1f"|format(scene.duration) }}s)</div>
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
{% for obj in scene.objects[:6] %}
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

<div class="card">
<h2>Object tracks</h2>
<div class="track-grid">
{% for track in index.tracks %}
<div class="track-card" data-start="{{ track.start }}">
<div class="body">
<strong>{{ track.label }}</strong> · track {{ track.track_id }}
<div>{{ "%.1f"|format(track.start) }}s – {{ "%.1f"|format(track.end) }}s
({{ "%.1f"|format(track.duration) }}s)</div>
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

<div class="card">
<h2>Detected actions</h2>
{% for action in index.actions %}
<div class="result jump" data-start="{{ action.start }}">
<strong>{{ "%.1f"|format(action.start) }}s – {{ "%.1f"|format(action.end) }}s</strong>
<div>{{ action.label }}{% if action.scene_id %} · scene {{ action.scene_id }}{% endif %}</div>
</div>
{% endfor %}
{% if not index.actions %}
<p class="note">No timed actions were detected.</p>
{% endif %}
</div>

<div class="card">
<h2>Speech transcript</h2>
{% if index.asr_error %}
<p class="warn" id="asr-error">Speech transcription failed: {{ index.asr_error }}</p>
<button type="button" class="get-caption-btn" id="retry-asr">Retry speech transcription</button>
{% endif %}
{% for seg in index.transcript %}
<div class="transcript-line jump" data-start="{{ seg.start }}">
<strong>{{ "%.1f"|format(seg.start) }}s</strong> — {{ seg.text }}
</div>
{% endfor %}
{% if not index.transcript and not index.asr_error %}
<p class="note">No speech was detected.</p>
{% endif %}
</div>
{% endif %}
</div>

<script>
(function () {
  const player = document.getElementById("player");
  const marker = document.getElementById("timeline-marker");
  const timeline = document.getElementById("timeline");
  const videoId = {{ (index.video_id if index else "")|tojson }};
  const jobId = {{ (job_id if job_id else "")|tojson }};

  if (jobId) {
    const statusEl = document.getElementById("job-status");
    function pollJob() {
      fetch("/video/job/" + jobId).then(function (r) { return r.json(); }).then(function (data) {
        if (statusEl) statusEl.textContent = data.message || data.status || "Working…";
        if (data.status === "done" && data.video_id) {
          window.location.href = "/video?id=" + data.video_id;
          return;
        }
        if (data.status === "error") {
          if (statusEl) {
            statusEl.className = "warn";
            statusEl.textContent = data.message || "Analysis failed.";
          }
          return;
        }
        setTimeout(pollJob, 1500);
      }).catch(function () { setTimeout(pollJob, 2500); });
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
        body: JSON.stringify({ video_id: videoId })
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
    player.currentTime = Math.max(0, Number(seconds) || 0);
    player.play();
  }

  document.querySelectorAll("[data-start]").forEach(function (el) {
    el.addEventListener("click", function () {
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
    }).then(function (r) { return r.json(); }).then(function (data) {
      if (data.error) {
        summary.textContent = data.error;
        box.innerHTML = "";
        return;
      }
      let extra = "";
      if (data.phrase_count !== undefined) {
        extra = " Speech hits: " + data.phrase_count + ". Visual hits: " + data.visual_count + ".";
      }
      summary.textContent = data.count + " result(s) for \"" + data.query + "\"." + extra;
      box.innerHTML = "";
      (data.results || []).forEach(function (item) {
        const div = document.createElement("div");
        div.className = "result";
        div.innerHTML = "<strong>" + item.label + "</strong> · " +
          (item.title || "") + "<div class='note'>" + (item.evidence || "") + "</div>";
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
</script>
</body>
</html>
"""
