VisualRAG — Word/Phrase Speech Frequency Search Fix

Files in this patch:
- video_search.py: counts every exact occurrence of a word, phrase, or sentence in the saved transcript; prefers Whisper word timestamps; groups nearby hits into readable time ranges; uses Unicode-aware normalization for non-English transcripts.
- video_ui.py: displays the total occurrence count, grouped time ranges with repeat counts, and a clickable result for each occurrence.

Install:
1. Stop Flask with Ctrl+C.
2. Back up the current video_search.py and video_ui.py in your project folder.
3. Copy these two files from this ZIP into the same folder as app.py, replacing the old copies.
4. Restart using the same virtual environment and command you normally use (for example: venv\\Scripts\\activate, then python app.py).
5. Open an already analyzed video and choose "Word / phrase (speech)". Enter a word, phrase, or sentence.

Important:
- The feature searches the transcript already saved in the video index. If transcription is empty or failed, there is no reliable speech text to search; run the app's Transcribe action first or re-analyze after fixing ASR.
- Exact word-level timestamps are used when Whisper saved word timings. If only transcript segments exist, timestamps are estimated within each segment and the UI labels them as estimated.
- Search is case-insensitive and punctuation-insensitive. Exact word/phrase matching is used, not semantic similarity.
