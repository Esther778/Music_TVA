from __future__ import annotations

import html
from pathlib import Path
import re
import subprocess
import urllib.request

import functional_section_pipeline as pipeline

from .lyric_search_agent import LyricSearchAgent
from .models import LyricSourceReport


class LyricSourceAgent:
    """Resolve usable time-aligned lyric lines.

    The agent prefers an explicit local timed-lyrics CSV, then a reusable
    Whisper transcript, and only runs Whisper when no timed source exists.
    Web lyric search belongs here later, but should store only derived timing
    and evidence snippets in outputs.
    """

    def run(
        self,
        audio: Path,
        work_dir: Path,
        model: str,
        language: str,
        reuse_transcript: bool,
        lyrics_csv: Path | None = None,
        lyrics_file: Path | None = None,
        lyrics_url: str = "",
        track_name: str = "",
        artist_name: str = "",
        duration_seconds: float | None = None,
    ) -> tuple[list[pipeline.LyricLine], LyricSourceReport]:
        if lyrics_url:
            lines = self._read_lyrics_url(lyrics_url)
            if lines:
                return lines, LyricSourceReport(
                    transcript_csv=Path(f"url:{lyrics_url}"),
                    source="web_synced_lyrics",
                    line_count=len(lines),
                    note="使用网页检索到的同步歌词，不落地保存完整歌词",
                )

        if track_name:
            result = LyricSearchAgent().search(track_name, artist_name, duration_seconds)
            if result is not None:
                return result.lines, LyricSourceReport(
                    transcript_csv=Path(f"lrclib:{result.matched_artist}-{result.matched_track}"),
                    source=result.source,
                    line_count=len(result.lines),
                    note=result.note,
                )

        if lyrics_file:
            lines = self._read_timed_lyrics_file(lyrics_file)
            return lines, LyricSourceReport(
                lyrics_file,
                "local_timed_lyrics_file",
                len(lines),
                "使用外部检索或人工提供的时间轴歌词文件",
            )

        transcript_csv = lyrics_csv or work_dir / f"{audio.stem}_whisper_segments.csv"
        source = "local_timed_lyrics_csv" if lyrics_csv else "whisper_transcript"

        if lyrics_csv is None and (not reuse_transcript or not transcript_csv.exists()):
            pipeline.run_transcription(audio, transcript_csv, model, language)
            source = "whisper_transcribed_now"

        lines = pipeline.read_transcript(transcript_csv)
        note = "使用本地时间轴歌词" if lyrics_csv else "使用 Whisper 句级时间轴"
        return lines, LyricSourceReport(transcript_csv, source, len(lines), note)

    def _read_timed_lyrics_file(self, path: Path) -> list[pipeline.LyricLine]:
        if path.suffix.lower() == ".csv":
            return pipeline.read_transcript(path)
        if path.suffix.lower() == ".lrc":
            return self._read_lrc(path)
        raise ValueError(f"Unsupported lyrics file type: {path.suffix}")

    def _read_lrc(self, path: Path) -> list[pipeline.LyricLine]:
        timestamp_pattern = re.compile(r"\[(\d{1,2}):(\d{2}(?:\.\d{1,3})?)\]")
        timed_rows: list[tuple[float, str]] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            matches = list(timestamp_pattern.finditer(raw_line))
            if not matches:
                continue
            text = timestamp_pattern.sub("", raw_line).strip()
            text = pipeline.clean_text(text)
            if not text or not pipeline.has_cjk(text) or pipeline.is_metadata_line(text):
                continue
            for match in matches:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                timed_rows.append((minutes * 60 + seconds, text))
        timed_rows.sort(key=lambda item: item[0])
        lines: list[pipeline.LyricLine] = []
        for index, (start, text) in enumerate(timed_rows):
            if index + 1 < len(timed_rows):
                end = max(start + 0.5, timed_rows[index + 1][0])
            else:
                end = start + 4.0
            lines.append(pipeline.LyricLine(start, end, text))
        return lines

    def _read_lyrics_url(self, url: str) -> list[pipeline.LyricLine]:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "codex-section-segmentation/0.1"})
            with urllib.request.urlopen(request, timeout=20) as response:
                text = response.read().decode("utf-8", errors="replace")
        except Exception:
            completed = subprocess.run(
                ["curl", "-L", url],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            text = completed.stdout
        match = re.search(r'id="lyrics-lyric"[^>]*>(.*?)</div>', text, flags=re.S)
        if not match:
            return []
        lyric_text = html.unescape(re.sub(r"<[^>]+>", "", match.group(1)))
        return self._parse_lrc_text(lyric_text)

    def _parse_lrc_text(self, text: str) -> list[pipeline.LyricLine]:
        timestamp_pattern = re.compile(r"\[(\d{1,2}):(\d{2}(?:\.\d{1,3})?)\]")
        timed_rows: list[tuple[float, str]] = []
        for raw_line in text.splitlines():
            matches = list(timestamp_pattern.finditer(raw_line))
            if not matches:
                continue
            lyric = timestamp_pattern.sub("", raw_line).strip()
            lyric = pipeline.clean_text(lyric)
            if not lyric or not pipeline.has_cjk(lyric) or pipeline.is_metadata_line(lyric):
                continue
            for match in matches:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                timed_rows.append((minutes * 60 + seconds, lyric))
        timed_rows.sort(key=lambda item: item[0])
        lines: list[pipeline.LyricLine] = []
        for index, (start, lyric) in enumerate(timed_rows):
            if index + 1 < len(timed_rows):
                end = max(start + 0.5, timed_rows[index + 1][0])
            else:
                end = start + 4.0
            lines.append(pipeline.LyricLine(start, end, lyric))
        return lines
