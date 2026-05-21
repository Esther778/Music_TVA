from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

import functional_section_pipeline as pipeline


@dataclass
class SearchLyricsResult:
    lines: list[pipeline.LyricLine]
    source: str
    matched_track: str
    matched_artist: str
    note: str


class LyricSearchAgent:
    """Fetch synchronized lyrics before falling back to transcription."""

    base_url = "https://lrclib.net/api"

    def search(
        self,
        track_name: str,
        artist_name: str = "",
        duration_seconds: float | None = None,
    ) -> SearchLyricsResult | None:
        queries = self._candidate_queries(track_name, artist_name, duration_seconds)
        for endpoint, params in queries:
            payload = self._request_json(endpoint, params)
            if payload is None:
                continue
            candidates = payload if isinstance(payload, list) else [payload]
            for candidate in candidates:
                result = self._candidate_to_result(candidate)
                if result is not None:
                    return result
        return None

    def _candidate_queries(
        self,
        track_name: str,
        artist_name: str,
        duration_seconds: float | None,
    ) -> list[tuple[str, dict[str, str]]]:
        queries: list[tuple[str, dict[str, str]]] = []
        exact: dict[str, str] = {"track_name": track_name}
        if artist_name:
            exact["artist_name"] = artist_name
        if duration_seconds:
            exact["duration"] = str(int(round(duration_seconds)))
        queries.append(("get", exact))

        search: dict[str, str] = {"track_name": track_name}
        if artist_name:
            search["artist_name"] = artist_name
        if duration_seconds:
            search["duration"] = str(int(round(duration_seconds)))
        queries.append(("search", search))

        query_text = f"{artist_name} {track_name}".strip()
        if query_text:
            queries.append(("search", {"q": query_text}))
            queries.append(("search", {"query": query_text}))
        return queries

    def _request_json(self, endpoint: str, params: dict[str, str]) -> object | None:
        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(
            f"{self.base_url}/{endpoint}?{query}",
            headers={"User-Agent": "codex-section-segmentation/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                if response.status >= 400:
                    return None
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            return None

    def _candidate_to_result(self, candidate: object) -> SearchLyricsResult | None:
        if not isinstance(candidate, dict):
            return None
        synced = candidate.get("syncedLyrics") or candidate.get("synced_lyrics")
        if not isinstance(synced, str) or not synced.strip():
            return None
        lines = self._parse_lrc_text(synced)
        if len(lines) < 4:
            return None
        return SearchLyricsResult(
            lines=lines,
            source="lrclib_synced_lyrics",
            matched_track=str(candidate.get("trackName") or candidate.get("track_name") or ""),
            matched_artist=str(candidate.get("artistName") or candidate.get("artist_name") or ""),
            note="LRCLIB synchronized lyrics used before Whisper fallback",
        )

    def _parse_lrc_text(self, text: str) -> list[pipeline.LyricLine]:
        import re

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
