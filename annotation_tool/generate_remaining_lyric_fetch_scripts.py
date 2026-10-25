#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
from pathlib import Path


LRCLIB_SEARCH = "https://lrclib.net/api/search?track_name={track}&artist_name={artist}"
NETEASE_SEARCH = "https://music.163.com/api/search/get/web?csrf_token=&type=1&offset=0&total=true&limit=10&s={query}"
NETEASE_LYRIC = "https://music.163.com/api/song/lyric?id={netease_id}&lv=1&kv=1&tv=-1"

EXTRA_QUERIES = {
    "life_should_face_light": [
        ("生活总该迎着光亮", "周深"),
        ("生活总该迎着光亮", "周深 宋祖儿 唐艺昕 毛晓彤 周翊然"),
        ("生活总该迎着光亮", ""),
    ],
    "local_048c4f7679cf": [
        ("别回头", "PAMajor"),
        ("别回头", "PAMajor INKK"),
        ("别回头", ""),
    ],
    "local_45be7a6e6f4f": [
        ("铃芽之旅", "周深"),
        ("铃芽之旅 中文版", "周深"),
        ("铃芽之旅", ""),
    ],
    "local_bfd33bfd9e12": [
        ("New Boy", "房东的猫"),
        ("New Boy", "朴树"),
        ("New Boy", ""),
    ],
    "local_d0f48fccda7c": [
        ("没那么简单", "关喆"),
        ("没那么简单", "黄小琥"),
        ("没那么简单", ""),
    ],
    "local_db974f892f9b": [
        ("花开忘忧", "周深"),
        ("花开忘忧", ""),
    ],
    "local_ef6d5a9855b1": [
        ("遇见", "孙燕姿"),
        ("遇见", ""),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate slower multi-query fetch scripts for remaining lyric misses.")
    parser.add_argument("--misses", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    songs = list(csv.DictReader(open(args.misses, encoding="utf-8")))
    write_lrclib_script(out_dir / "fetch_remaining_lrclib_variants.sh", out_dir / "lrclib_variants", songs)
    write_netease_search_script(out_dir / "fetch_remaining_netease_search.sh", out_dir / "netease_search_variants", songs)
    write_netease_lyrics_script(out_dir / "fetch_remaining_netease_lyrics.sh", out_dir / "netease_search_variants", out_dir / "netease_lyrics_variants")
    print(f"scripts written to {out_dir}")


def write_lrclib_script(path: Path, cache_dir: Path, songs: list[dict]) -> None:
    lines = ["#!/usr/bin/env zsh", "set -uo pipefail", f"mkdir -p {sh(str(cache_dir))}", "ok=0", "fail=0"]
    tasks = []
    for song in songs:
        for idx, (track, artist) in enumerate(EXTRA_QUERIES.get(song["id"], [(song["title"], song["artist"])]), start=1):
            url = LRCLIB_SEARCH.format(track=urllib.parse.quote(track), artist=urllib.parse.quote(artist))
            out = cache_dir / f"{song['id']}__q{idx}.json"
            tasks.append((song, idx, url, out))
    total = len(tasks)
    for i, (song, idx, url, out) in enumerate(tasks, start=1):
        progress = f"[{i}/{total}] LRCLIB {song['title']} q{idx}"
        lines += [
            f"echo {sh(progress)}",
            f"if curl -sS --connect-timeout 8 --max-time 25 --retry 3 --retry-all-errors --retry-delay 2 -A 'music-section-qc/1.0' {sh(url)} -o {sh(str(out))}; then",
            "  ok=$((ok+1))",
            "else",
            "  fail=$((fail+1))",
            f"  echo '[]' > {sh(str(out))}",
            "fi",
            "sleep 1",
        ]
    lines.append('echo "Remaining LRCLIB variants complete. ok=${ok} fail=${fail}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_netease_search_script(path: Path, cache_dir: Path, songs: list[dict]) -> None:
    lines = ["#!/usr/bin/env zsh", "set -uo pipefail", f"mkdir -p {sh(str(cache_dir))}", "ok=0", "fail=0"]
    tasks = []
    for song in songs:
        for idx, (track, artist) in enumerate(EXTRA_QUERIES.get(song["id"], [(song["title"], song["artist"])]), start=1):
            query = f"{track} {artist}".strip()
            url = NETEASE_SEARCH.format(query=urllib.parse.quote(query))
            out = cache_dir / f"{song['id']}__q{idx}.json"
            tasks.append((song, idx, url, out))
    total = len(tasks)
    for i, (song, idx, url, out) in enumerate(tasks, start=1):
        progress = f"[{i}/{total}] NetEase search {song['title']} q{idx}"
        lines += [
            f"echo {sh(progress)}",
            f"if curl -sS --connect-timeout 8 --max-time 25 --retry 3 --retry-all-errors --retry-delay 3 -A 'Mozilla/5.0' -H 'Referer: https://music.163.com/' {sh(url)} -o {sh(str(out))}; then",
            "  ok=$((ok+1))",
            "else",
            "  fail=$((fail+1))",
            f"  echo '{{}}' > {sh(str(out))}",
            "fi",
            "sleep 2",
        ]
    lines.append('echo "Remaining NetEase search complete. ok=${ok} fail=${fail}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_netease_lyrics_script(path: Path, search_dir: Path, lyric_dir: Path) -> None:
    lines = [
        "#!/usr/bin/env zsh",
        "set -euo pipefail",
        f"mkdir -p {sh(str(lyric_dir))}",
        "tmp_file=$(mktemp)",
        "python3 - <<'PY' > ${tmp_file}",
        "import json, pathlib",
        f"search_dir = pathlib.Path({str(search_dir)!r})",
        f"lyric_dir = pathlib.Path({str(lyric_dir)!r})",
        "for path in sorted(search_dir.glob('*.json')):",
        "    try: data = json.loads(path.read_text(encoding='utf-8'))",
        "    except Exception: continue",
        "    songs = (data.get('result') or {}).get('songs') or []",
        "    for item in songs[:10]:",
        "        netease_id = item.get('id')",
        "        if not netease_id: continue",
        "        out = lyric_dir / f'{path.stem}__{netease_id}.json'",
        "        url = " + repr(NETEASE_LYRIC) + ".format(netease_id=netease_id)",
        "        print(f'{path.stem}\\t{netease_id}\\t{url}\\t{out}')",
        "PY",
        "total=$(wc -l < ${tmp_file} | tr -d ' ')",
        "ok=0",
        "fail=0",
        "i=0",
        "while IFS=$'\\t' read -r song_key netease_id url out; do",
        "  i=$((i+1))",
        '  echo "[${i}/${total}] ${song_key} netease=${netease_id}"',
        '  if curl -sS --connect-timeout 8 --max-time 25 --retry 3 --retry-all-errors --retry-delay 3 -A "Mozilla/5.0" -H "Referer: https://music.163.com/" "${url}" -o "${out}"; then',
        "    ok=$((ok+1))",
        "  else",
        "    fail=$((fail+1))",
        "    echo '{}' > \"${out}\"",
        "  fi",
        "  sleep 1",
        "done < ${tmp_file}",
        "rm -f ${tmp_file}",
        'echo "Remaining NetEase lyrics complete. ok=${ok} fail=${fail}"',
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sh(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    main()
