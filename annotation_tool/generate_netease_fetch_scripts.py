#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
from pathlib import Path


SEARCH_URL = "https://music.163.com/api/search/get/web?csrf_token=&type=1&offset=0&total=true&limit=8&s={query}"
LYRIC_URL = "https://music.163.com/api/song/lyric?id={netease_id}&lv=1&kv=1&tv=-1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate curl scripts for NetEase lyric fallback cache.")
    parser.add_argument("--misses", required=True, help="CSV of songs still missing lyric corrections.")
    parser.add_argument("--out-dir", required=True, help="Output directory for scripts and cache.")
    parser.add_argument("--search-script", default="fetch_netease_search_cache.sh")
    parser.add_argument("--lyric-script", default="fetch_netease_lyric_cache.sh")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    search_dir = out_dir / "netease_search"
    lyric_dir = out_dir / "netease_lyrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    songs = list(csv.DictReader(open(args.misses, encoding="utf-8")))
    write_search_script(out_dir / args.search_script, search_dir, songs)
    write_lyric_script(out_dir / args.lyric_script, search_dir, lyric_dir)
    print(f"search script: {out_dir / args.search_script}")
    print(f"lyric script: {out_dir / args.lyric_script}")


def write_search_script(path: Path, search_dir: Path, songs: list[dict]) -> None:
    lines = [
        "#!/usr/bin/env zsh",
        "set -uo pipefail",
        f"mkdir -p {shell_quote(str(search_dir))}",
        "ok=0",
        "fail=0",
    ]
    total = len(songs)
    for index, song in enumerate(songs, start=1):
        query = f"{song['title']} {song['artist']}".strip()
        url = SEARCH_URL.format(query=urllib.parse.quote(query))
        out = search_dir / f"{song['id']}.json"
        label = f"[{index}/{total}] {song['title']} - {song['artist']}"
        lines.extend(
            [
                f"echo {shell_quote(label)}",
                f"if curl -sS --connect-timeout 8 --max-time 20 --retry 3 --retry-all-errors --retry-delay 2 {shell_quote(url)} -o {shell_quote(str(out))}; then",
                "  ok=$((ok+1))",
                "else",
                "  fail=$((fail+1))",
                f"  echo '{{}}' > {shell_quote(str(out))}",
                "fi",
            ]
        )
    lines.append('echo "NetEase search cache fetch complete. ok=${ok} fail=${fail}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lyric_script(path: Path, search_dir: Path, lyric_dir: Path) -> None:
    lines = [
        "#!/usr/bin/env zsh",
        "set -euo pipefail",
        f"mkdir -p {shell_quote(str(lyric_dir))}",
        "python3 - <<'PY'",
        "import json, pathlib",
        f"search_dir = pathlib.Path({str(search_dir)!r})",
        f"lyric_dir = pathlib.Path({str(lyric_dir)!r})",
        "commands = []",
        "for path in sorted(search_dir.glob('*.json')):",
        "    try:",
        "        data = json.loads(path.read_text(encoding='utf-8'))",
        "    except Exception:",
        "        continue",
        "    songs = (data.get('result') or {}).get('songs') or []",
        "    seen = set()",
        "    for item in songs[:8]:",
        "        netease_id = item.get('id')",
        "        if not netease_id or netease_id in seen:",
        "            continue",
        "        seen.add(netease_id)",
        "        out = lyric_dir / f'{path.stem}__{netease_id}.json'",
        "        url = " + repr(LYRIC_URL) + ".format(netease_id=netease_id)",
        "        commands.append((path.stem, netease_id, url, str(out)))",
        "print(len(commands))",
        "for song_id, netease_id, url, out in commands:",
        "    print(f'{song_id}\\t{netease_id}\\t{url}\\t{out}')",
        "PY",
        "",
        "tmp_file=$(mktemp)",
        "python3 - <<'PY' > ${tmp_file}",
        "import json, pathlib",
        f"search_dir = pathlib.Path({str(search_dir)!r})",
        f"lyric_dir = pathlib.Path({str(lyric_dir)!r})",
        "for path in sorted(search_dir.glob('*.json')):",
        "    try:",
        "        data = json.loads(path.read_text(encoding='utf-8'))",
        "    except Exception:",
        "        continue",
        "    songs = (data.get('result') or {}).get('songs') or []",
        "    seen = set()",
        "    for item in songs[:8]:",
        "        netease_id = item.get('id')",
        "        if not netease_id or netease_id in seen:",
        "            continue",
        "        seen.add(netease_id)",
        "        out = lyric_dir / f'{path.stem}__{netease_id}.json'",
        "        url = " + repr(LYRIC_URL) + ".format(netease_id=netease_id)",
        "        print(f'{path.stem}\\t{netease_id}\\t{url}\\t{out}')",
        "PY",
        "total=$(wc -l < ${tmp_file} | tr -d ' ')",
        "ok=0",
        "fail=0",
        "i=0",
        "while IFS=$'\\t' read -r song_id netease_id url out; do",
        "  i=$((i+1))",
        '  echo "[${i}/${total}] ${song_id} netease=${netease_id}"',
        '  if curl -sS --connect-timeout 8 --max-time 20 --retry 3 --retry-all-errors --retry-delay 2 "${url}" -o "${out}"; then',
        "    ok=$((ok+1))",
        "  else",
        "    fail=$((fail+1))",
        "    echo '{}' > \"${out}\"",
        "  fi",
        "done < ${tmp_file}",
        "rm -f ${tmp_file}",
        'echo "NetEase lyric cache fetch complete. ok=${ok} fail=${fail}"',
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def shell_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


if __name__ == "__main__":
    main()
