#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


SUPABASE_URL = "https://itcarzyhlkiovskfdqut.supabase.co"
SUPABASE_KEY = "sb_publishable_IG4iiZuIDAoxhzZSPZnRRA_xSl0VRfN"

TRADITIONAL_TO_SIMPLIFIED = str.maketrans(
    {
        "妳": "你",
        "沒": "没",
        "聽": "听",
        "對": "对",
        "愛": "爱",
        "錯": "错",
        "著": "着",
        "說": "说",
        "話": "话",
        "裡": "里",
        "騙": "骗",
        "會": "会",
        "懂": "懂",
        "從": "从",
        "後": "后",
        "願": "愿",
        "變": "变",
        "張": "张",
        "雙": "双",
        "護": "护",
        "們": "们",
        "樂": "乐",
        "結": "结",
        "寫": "写",
        "風": "风",
        "長": "长",
        "髮": "发",
        "淚": "泪",
        "歲": "岁",
        "這": "这",
        "誰": "谁",
        "離": "离",
        "與": "与",
        "負": "负",
        "無": "无",
        "為": "为",
        "還": "还",
        "過": "过",
        "處": "处",
        "紅": "红",
        "間": "间",
        "盡": "尽",
        "夢": "梦",
        "雲": "云",
        "給": "给",
        "來": "来",
        "將": "将",
        "現": "现",
        "讓": "让",
        "別": "别",
        "嗎": "吗",
        "輕": "轻",
        "舊": "旧",
        "時": "时",
        "畫": "画",
        "記": "记",
        "憶": "忆",
        "樣": "样",
        "萬": "万",
        "聲": "声",
        "邊": "边",
        "點": "点",
        "飛": "飞",
        "屬": "属",
        "際": "际",
        "層": "层",
        "闊": "阔",
        "涼": "凉",
        "綻": "绽",
        "臺": "台",
        "詞": "词",
        "曲": "曲",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full-song lyric text corrections from LRCLIB synced lyrics.")
    parser.add_argument("--songs", required=True, help="songs JSON exported from Supabase.")
    parser.add_argument("--lyrics", nargs="+", required=True, help="lyric_lines JSON page files exported from Supabase.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--apply-script", default="apply_lrclib_lyric_corrections.sh")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--min-match-lines", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "lrclib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    songs = read_json(Path(args.songs))
    lyric_rows = []
    for path in args.lyrics:
        lyric_rows.extend(read_json(Path(path)))

    rows_by_song: dict[str, list[dict]] = defaultdict(list)
    for row in lyric_rows:
        rows_by_song[row["song_id"]].append(row)
    for rows in rows_by_song.values():
        rows.sort(key=lambda item: (int(item["line_index"]), float(item["start_time_sec"])))

    corrections = []
    misses = []
    reports = []
    for song in songs:
        rows = rows_by_song.get(song["id"], [])
        if not rows:
            misses.append({**song, "reason": "no_database_lyric_rows"})
            continue
        result = find_lrclib_result(song, cache_dir)
        if not result:
            misses.append({**song, "reason": "lrclib_no_match"})
            continue
        synced = parse_lrc(result.get("syncedLyrics") or "")
        if len(synced) < args.min_match_lines:
            misses.append({**song, "reason": "lrclib_no_synced_lyrics", "lrclib_id": result.get("id")})
            continue
        generated = align_rows(rows, synced)
        changed = 0
        for row, corrected in generated:
            current = clean_text(row.get("text") or "")
            if corrected and corrected != current:
                corrections.append(
                    {
                        "song_id": row["song_id"],
                        "line_index": int(row["line_index"]),
                        "start_time_sec": row["start_time_sec"],
                        "end_time_sec": row["end_time_sec"],
                        "current_text": row.get("text") or "",
                        "corrected_text": corrected,
                        "source": f"lrclib:{result.get('id')}",
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
                "lrclib_id": result.get("id"),
                "lrclib_track": result.get("trackName"),
                "lrclib_artist": result.get("artistName"),
                "database_lines": len(rows),
                "lrc_lines": len(synced),
                "changed_lines": changed,
            }
        )

    write_csv(
        out_dir / "lrclib_corrections.csv",
        corrections,
        ["song_id", "line_index", "start_time_sec", "end_time_sec", "current_text", "corrected_text", "source", "title", "artist"],
    )
    write_csv(
        out_dir / "lrclib_misses.csv",
        misses,
        ["id", "title", "artist", "audio_path", "status", "duration_sec", "reason", "lrclib_id"],
    )
    write_csv(
        out_dir / "lrclib_report.csv",
        reports,
        ["song_id", "title", "artist", "lrclib_id", "lrclib_track", "lrclib_artist", "database_lines", "lrc_lines", "changed_lines"],
    )
    write_apply_script(out_dir / args.apply_script, corrections)
    print(f"matched songs: {len(reports)}")
    print(f"missed songs: {len(misses)}")
    print(f"correction rows: {len(corrections)}")
    print(f"apply script: {out_dir / args.apply_script}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_lrclib_result(song: dict, cache_dir: Path) -> dict | None:
    title = normalize_title(song["title"])
    artist = normalize_artist(song.get("artist") or "")
    cache_path = cache_dir / f"{song['id']}.json"
    if cache_path.exists():
        results = read_json(cache_path)
    else:
        query = urllib.parse.urlencode({"track_name": title, "artist_name": artist})
        url = f"https://lrclib.net/api/search?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": "music-section-qc/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                results = json.loads(response.read().decode("utf-8"))
        except Exception as error:
            cache_path.write_text(json.dumps({"error": str(error)}, ensure_ascii=False, indent=2), encoding="utf-8")
            return None
        cache_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if isinstance(results, dict):
        return None
    duration = float(song.get("duration_sec") or 0)
    wants_chinese = has_han(title) or has_han(artist)
    scored = []
    for item in results:
        synced_lyrics = item.get("syncedLyrics") or ""
        if not synced_lyrics:
            continue
        if wants_chinese and not is_chinese_lyric_candidate(synced_lyrics):
            continue
        track = normalize_key(item.get("trackName") or item.get("name") or "")
        item_artist = normalize_key(item.get("artistName") or "")
        title_score = 1 if normalize_key(title) in track or track in normalize_key(title) else 0
        artist_score = 1 if normalize_key(artist) and (normalize_key(artist) in item_artist or item_artist in normalize_key(artist)) else 0
        synced_score = 1
        duration_delta = abs(float(item.get("duration") or 0) - duration) if duration else 0
        scored.append((title_score, artist_score, synced_score, -duration_delta, item))
    scored.sort(reverse=True, key=lambda row: row[:4])
    best = scored[0][4] if scored else None
    if not best or not best.get("syncedLyrics"):
        return None
    return best


def has_han(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", value))


def is_chinese_lyric_candidate(value: str) -> bool:
    han_count = len(re.findall(r"[\u4e00-\u9fff]", value))
    latin_count = len(re.findall(r"[A-Za-z]", value))
    if han_count < 12:
        return False
    return han_count >= latin_count * 0.7


def normalize_title(value: str) -> str:
    value = re.sub(r"（.*?）|\(.*?\)|《|》", "", value)
    return value.strip()


def normalize_artist(value: str) -> str:
    return re.split(r"[/、，,]", value)[0].strip()


def normalize_key(value: str) -> str:
    return re.sub(r"\s+", "", clean_text(value)).lower()


def parse_lrc(text: str) -> list[tuple[float, str]]:
    rows = []
    pattern = re.compile(r"\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\]")
    for raw in text.splitlines():
        matches = list(pattern.finditer(raw))
        lyric = pattern.sub("", raw).strip()
        lyric = clean_text(lyric)
        for match in matches:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            frac = match.group(3) or "0"
            timestamp = minutes * 60 + seconds + int(frac.ljust(3, "0")[:3]) / 1000
            rows.append((timestamp, lyric))
    return sorted((time, lyric) for time, lyric in rows if lyric)


def align_rows(db_rows: list[dict], synced: list[tuple[float, str]]) -> list[tuple[dict, str]]:
    first_lrc = synced[0][0]
    last_lrc = synced[-1][0]
    generated = []
    for row in db_rows:
        start = float(row["start_time_sec"])
        end = float(row["end_time_sec"])
        if end <= first_lrc:
            generated.append((row, "前奏"))
            continue
        if start >= last_lrc + 8:
            generated.append((row, "尾奏"))
            continue
        texts = [lyric for time, lyric in synced if start - 0.35 <= time < end + 0.35]
        if not texts:
            current = clean_text(row.get("text") or "")
            if current in {"前奏", "尾奏", "间奏"}:
                generated.append((row, current))
            elif start < first_lrc:
                generated.append((row, "前奏"))
            elif start > last_lrc:
                generated.append((row, "尾奏"))
            else:
                generated.append((row, "间奏"))
            continue
        deduped = []
        for text in texts:
            if text and (not deduped or deduped[-1] != text):
                deduped.append(text)
        generated.append((row, " ".join(deduped)))
    return generated


def clean_text(value: str) -> str:
    text = value.translate(TRADITIONAL_TO_SIMPLIFIED)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
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
        'echo "Applying LRCLIB lyric text corrections..."',
    ]
    for row in corrections:
        payload = json.dumps({"text": row["corrected_text"]}, ensure_ascii=False)
        lines.append(f"patch_text {shell_quote(row['song_id'])} {int(row['line_index'])} {shell_quote(payload)}")
    lines.append('echo "LRCLIB lyric correction complete."')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def shell_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


if __name__ == "__main__":
    main()
