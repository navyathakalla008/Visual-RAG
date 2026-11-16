from pathlib import Path


def test_transcript_lines_include_playback_boundaries():
    html = Path("video_ui.py").read_text(encoding="utf-8")
    assert 'data-start="{{ seg.start }}" data-end="{{ seg.end }}"' in html
    assert 'player.currentTime = Math.min(target, Math.max(0, duration))' in html
    assert 'player.play()' in html
    assert 'id="player-feedback" class="note" role="status" aria-live="polite" hidden' in html
    assert 'Playback could not start. Use the video controls to retry.' in html
    assert 'window.scrollTo' in html
    assert 'behavior: "auto"' in html
    assert 'line.classList.toggle("active"' in html


def test_video_entries_seek_by_click_and_keyboard():
    html = Path("video_ui.py").read_text(encoding="utf-8")
    assert 'class="scene-card" data-start="{{ scene.start }}" tabindex="0"' in html
    assert 'class="track-card" data-start="{{ track.start }}" tabindex="0" role="button"' in html
    assert 'class="result jump" data-start="{{ action.start }}" tabindex="0" role="button"' in html
    assert 'class="transcript-line jump" data-start="{{ seg.start }}" data-end="{{ seg.end }}"' in html
    assert 'event.key !== "Enter" && event.key !== " "' in html
    assert '.scene-card[data-start], .track-card[data-start], .result.jump[data-start], .transcript-line[data-start]' in html
    assert 'event.target.closest(".scene-character-details, .video-characters-details, summary")' in html
    assert 'event.target.closest("details") return' not in html
    assert 'window.requestAnimationFrame(scrollToPlayer)' not in html
    assert 'window.setTimeout(scrollToPlayer, 120)' not in html


if __name__ == "__main__":
    test_transcript_lines_include_playback_boundaries()
    test_video_entries_seek_by_click_and_keyboard()
    print("Video jump regression tests passed.")
