import app
from unittest.mock import patch

client = app.app.test_client()
with patch("app.list_video_indexes", wraps=app.list_video_indexes) as list_indexes:
    response = client.get("/video")
html = response.get_data(as_text=True)

assert response.status_code == 200
assert list_indexes.call_count == 1
assert html.index("<style>") < html.index("</style>") < html.index("<body>")
assert "Upload a video" in html
assert 'name="video"' in html
assert "Analyze video" in html
assert html.count("<script>") == 1
assert "loadCharacterGroups" in html
assert html.count('class="section-toggle"') >= 7
assert ".section-toggle::after" in html
assert ".collapsible-card[open] > .section-toggle::after" in html
assert "Only confidently matched character names are shown." in html
assert "Names mentioned in these scenes" not in html
library = app.list_video_indexes()
if library:
    with patch("app._video_character_groups") as character_groups:
        detail_response = client.get("/video?id=" + library[0]["video_id"])
    assert detail_response.status_code == 200
    detail_html = detail_response.get_data(as_text=True)
    assert detail_html.count('class="section-toggle"') == 8
    assert '<summary class="section-toggle"><h2>Scenes</h2></summary>' in detail_html
    assert '<summary class="section-toggle"><h2>Speech transcript</h2></summary>' in detail_html
    character_groups.assert_not_called()
print({
    "status": response.status_code,
    "bytes": len(html),
    "html_order_ok": True,
    "upload_form": True,
    "script_present": True,
    "library_scans": list_indexes.call_count,
})
