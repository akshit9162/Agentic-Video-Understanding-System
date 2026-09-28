import glob
import os
import shutil
import tempfile
import uuid

import yt_dlp

from celery_app import celery_app


@celery_app.task(bind=True)
def summarize_video_task(self, source_type, source_value, ratio=0.2):
    tmp_dir = tempfile.mkdtemp(prefix="video_sum_")
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", f"summary_{uuid.uuid4().hex}.mp4")

    try:
        if source_type == "upload":
            local_video = source_value
            self.update_state(state="PROGRESS", meta={"progress": 5, "stage": "Using uploaded video"})
        elif source_type == "youtube":
            self.update_state(state="PROGRESS", meta={"progress": 1, "stage": "Downloading YouTube video"})
            out_tmpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")

            def _download_hook(d):
                if d.get("status") == "downloading":
                    total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    downloaded = d.get("downloaded_bytes", 0)
                    if total > 0:
                        pct = int(min(100, (downloaded / total) * 100))
                        self.update_state(
                            state="PROGRESS",
                            meta={"progress": int(0.35 * pct), "stage": f"Downloading YouTube video ({pct}%)"},
                        )
                elif d.get("status") == "finished":
                    self.update_state(state="PROGRESS", meta={"progress": 35, "stage": "Download complete"})

            ydl_opts = {
                "format": "bestvideo+bestaudio/best",
                "merge_output_format": "mp4",
                "outtmpl": out_tmpl,
                "noplaylist": True,
                "quiet": True,
                "progress_hooks": [_download_hook],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([source_value])

            candidates = sorted(glob.glob(os.path.join(tmp_dir, "*.mp4")))
            if not candidates:
                raise RuntimeError("YouTube download did not produce an mp4 file.")
            local_video = candidates[-1]
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        def _progress_callback(value, message):
            pct = 35 + int(float(value) * 65)
            self.update_state(state="PROGRESS", meta={"progress": pct, "stage": message})

        # Imported here, not at module level: it loads CLIP, Whisper and torch, which only
        # the Celery worker needs. The API process imports this module too and stays light.
        from src.inference import run_inference

        result = run_inference(
            local_video,
            output_path,
            target_ratio=float(ratio),
            progress_callback=_progress_callback,
        )
        return {
            "video": result["video"],
            "plot": result["plot"],
            "speech_transcript": result.get("speech_transcript", ""),
            "video_hash": result.get("video_hash", ""),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
