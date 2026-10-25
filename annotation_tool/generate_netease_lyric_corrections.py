#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import generate_lrclib_lyric_corrections as lrclib


SUPABASE_URL = lrclib.SUPABASE_URL
SUPABASE_KEY = lrclib.SUPABASE_KEY


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lyric text corrections from NetEase LRC fallback cache.")
    parser.add_argument("--misses", required=True)
    parser.add_argument("--songs", required=True)
    parser.add_argument("--lyrics", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--lyric-dir", required=True)
    parser.add_argument("--apply-script", default="apply_netease_lyric_corrections.sh")
    parser.add_argument("--min-lrc-lines", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    search_dir = Path(args.search_dir)
    lyric_dir = Path(args.lyric_dir)
    candidates_path = out_dir / "netease_candidates.csv"
    if candidates_path.exists():
        candidates_path.unlink()

    all_songs = {row["id"]: row for row in lrclib.read_json(Path(args.songs))}
    targets = []
    for miss in csv.DictReader(open(args.misses, encoding="utf-8")):
        song = all_songs.get(miss["id"])
        if song:
            targets.append(song)

    lyric_rows = []
    for path in args.lyrics:
        lyric_rows.extend(lrclib.read_json(Path(path)))
    rows_by_song: dict[str, list[dict]] = defaultdict(list)
    for row in lyric_rows:
        rows_by_song[row["song_id"]].append(row)
    for rows in rows_by_song.values():
        rows.sort(key=lambda item: (int(item["line_index"]), float(item["start_time_sec"])))

    corrections = []
    reports = []
    misses = []
    for song in targets:
        rows = rows_by_song.get(song["id"], [])
        if not rows:
            misses.append({**song, "reason": "no_database_lyric_rows"})
            continue
        best, candidates = find_best_candidate(song, search_dir, lyric_dir, args.min_lrc_lines)
        write_candidate_rows(candidates_path, candidates, append=bool(reports or misses or corrections))
        if not best:
            misses.append({**song, "reason": "netease_no_high_confidence_match"})
            continue
        synced = lrclib.parse_lrc(best["lyric"])
        generated = lrclib.align_rows(rows, synced)
        changed = 0
        for row, corrected in generated:
            current = lrclib.clean_text(row.get("text") or "")
            if corrected and corrected != current:
                corrections.append(
                    {
                        "song_id": row["song_id"],
                        "line_index": int(row["line_index"]),
                        "start_time_sec": row["start_time_sec"],
                        "end_time_sec": row["end_time_sec"],
                        "current_text": row.get("text") or "",
                        "corrected_text": corrected,
                        "source": f"netease:{best['netease_id']}",
                        "title": song["title"],
                        "artist": song.get("artist") or "",
                    }
                )
                changed += 1
        reports.append(
            {
                "song_id": song["id"],
                "title": song["title"],
                "artist": song.get("artist") or "",
                "netease_id": best["netease_id"],
                "netease_name": best["name"],
                "netease_artists": best["artists"],
                "score": best["score"],
                "database_lines": len(rows),
                "lrc_lines": len(synced),
                "changed_lines": changed,
                "duration_delta_sec": best["duration_delta_sec"],
            }
        )

    lrclib.write_csv(
        out_dir / "netease_corrections.csv",
        corrections,
        ["song_id", "line_index", "start_time_sec", "end_time_sec", "current_text", "corrected_text", "source", "title", "artist"],
    )
    lrclib.write_csv(
        out_dir / "netease_report.csv",
        reports,
        ["song_id", "title", "artist", "netease_id", "netease_name", "netease_artists", "score", "database_lines", "lrc_lines", "changed_lines", "duration_delta_sec"],
    )
    lrclib.write_csv(
        out_dir / "netease_misses.csv",
        misses,
        ["id", "title", "artist", "audio_path", "status", "duration_sec", "reason"],
    )
    write_apply_script(out_dir / args.apply_script, corrections)
    print(f"matched songs: {len(reports)}")
    print(f"missed songs: {len(misses)}")
    print(f"correction rows: {len(corrections)}")
    print(f"apply script: {out_dir / args.apply_script}")


def find_best_candidate(song: dict, search_dir: Path, lyric_dir: Path, min_lrc_lines: int) -> tuple[dict | None, list[dict]]:
    search_path = search_dir / f"{song['id']}.json"
    if not search_path.exists():
        return None, []
    try:
        data = json.loads(search_path.read_text(encoding="utf-8"))
    except Exception:
        return None, []
    items = (data.get("result") or {}).get("songs") or []
    duration = float(song.get("duration_sec") or 0)
    target_title = normalize_song_title(song["title"])
    target_artist = lrclib.normalize_artist(song.get("artist") or "")
    rows = []
    for rank, item in enumerate(items[:8], start=1):
        netease_id = str(item.get("id") or "")
        lyric_path = lyric_dir / f"{song['id']}__{netease_id}.json"
        lyric = ""
        if lyric_path.exists():
            try:
                lyric_data = json.loads(lyric_path.read_text(encoding="utf-8"))
                lyric = (lyric_data.get("lrc") or {}).get("lyric") or ""
            except Exception:
                lyric = ""
        parsed = lrclib.parse_lrc(lyric)
        name = item.get("name") or ""
        artists = " / ".join(artist.get("name", "") for artist in item.get("artists", []))
        item_duration = float(item.get("duration") or 0) / 1000
        duration_delta = abs(item_duration - duration) if duration and item_duration else 9999.0
        title_score = title_match_score(target_title, name)
        artist_score = artist_match_score(target_artist, artists)
        artist_in_name = lrclib.normalize_key(target_artist) and lrclib.normalize_key(target_artist) in lrclib.normalize_key(name)
        artist_confident = artist_score >= 0.7 or artist_in_name
        chinese_ok = (not lrclib.has_han(target_title + target_artist)) or lrclib.is_chinese_lyric_candidate(lyric)
        if title_score >= 1 and artist_confident and len(parsed) >= min_lrc_lines:
            chinese_ok = True
        instrumental_penalty = 1 if is_instrumental_name(name) else 0
        lrc_score = min(len(parsed), 50) / 50
        duration_score = max(0, 1 - min(duration_delta, 30) / 30)
        score = title_score * 4 + artist_score * 2 + lrc_score * 2 + duration_score - instrumental_penalty * 3 - rank * 0.03
        valid = title_score >= 0.75 and artist_confident and len(parsed) >= min_lrc_lines and chinese_ok and not instrumental_penalty and duration_delta <= 45
        rows.append(
            {
                "song_id": song["id"],
                "target_title": song["title"],
                "target_artist": song.get("artist") or "",
                "netease_id": netease_id,
                "rank": rank,
                "name": name,
                "artists": artists,
                "lrc_lines": len(parsed),
                "duration_delta_sec": round(duration_delta, 2),
                "title_score": round(title_score, 2),
                "artist_score": round(artist_score, 2),
                "artist_in_name": bool(artist_in_name),
                "chinese_ok": chinese_ok,
                "instrumental_penalty": instrumental_penalty,
                "score": round(score, 3),
                "valid": valid,
                "lyric": lyric,
            }
        )
    valid_rows = [row for row in rows if row["valid"]]
    if not valid_rows:
        return None, rows
    best = sorted(valid_rows, key=lambda row: row["score"], reverse=True)[0]
    return best, rows


def normalize_song_title(value: str) -> str:
    value = lrclib.normalize_title(value)
    value = re.sub(r"(?i)\blive\b|中文版|原版|伴奏|和声|钢琴版|吉他|纯音乐|cover", "", value)
    value = re.sub(r"\s+", "", lrclib.clean_text(value)).lower()
    return value


def title_match_score(target: str, candidate: str) -> float:
    cand = normalize_song_title(candidate)
    if not target or not cand:
        return 0
    if target == cand:
        return 1
    if target in cand or cand in target:
        return 0.85
    return len(set(target) & set(cand)) / max(len(set(target)), 1) * 0.5


def artist_match_score(target: str, candidate: str) -> float:
    target_key = lrclib.normalize_key(target)
    cand_key = lrclib.normalize_key(candidate)
    if not target_key:
        return 0
    if target_key in cand_key or cand_key in target_key:
        return 1
    target_parts = [lrclib.normalize_key(part) for part in re.split(r"[/、，,]", target) if part.strip()]
    cand_parts = [lrclib.normalize_key(part) for part in re.split(r"[/、，,]", candidate) if part.strip()]
    if any(part and part in cand_key for part in target_parts):
        return 0.8
    if set(target_parts) & set(cand_parts):
        return 0.7
    return 0


def is_instrumental_name(value: str) -> bool:
    return bool(re.search(r"伴奏|纯音乐|钢琴|吉他|指弹|instrumental|karaoke", value, re.I))


def write_candidate_rows(path: Path, rows: list[dict], append: bool) -> None:
    fields = [
        "song_id", "target_title", "target_artist", "netease_id", "rank", "name", "artists",
        "lrc_lines", "duration_delta_sec", "title_score", "artist_score", "artist_in_name", "chinese_ok",
        "instrumental_penalty", "score", "valid",
    ]
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def write_apply_script(path: Path, corrections: list[dict]) -> None:
    lines = [
        "#!/usr/bin/env zsh",
        "set -euo pipefail",
        "",
        f"BASE={json.dumps(SUPABASE_URL + '/rest/v1', ensure_ascii=False)}",
        f"KEY={json.dumps(SUPABASE_KEY, ensure_ascii=False)}",
        "",
        "patch_text() {",
        '  local song_id="$1"',
        '  local line_index="$2"',
        '  local payload="$3"',
        "  curl -sS --retry 5 --retry-all-errors --retry-delay 2 \\",
        "    -X PATCH \\",
        '    -H "apikey: ${KEY}" \\',
        '    -H "Authorization: Bearer ${KEY}" \\',
        '    -H "Content-Type: application/json" \\',
        '    -H "Prefer: return=minimal" \\',
        '    --data-binary "${payload}" \\',
        '    "${BASE}/lyric_lines?song_id=eq.${song_id}&line_index=eq.${line_index}"',
        "}",
        "",
        'echo "Applying NetEase lyric text corrections..."',
    ]
    for row in corrections:
        payload = json.dumps({"text": row["corrected_text"]}, ensure_ascii=False)
        lines.append(f"patch_text {lrclib.shell_quote(row['song_id'])} {int(row['line_index'])} {lrclib.shell_quote(payload)}")
    lines.append('echo "NetEase lyric correction complete."')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
