#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/data/blind_validation/stage1/raw"
mkdir -p "$OUT_DIR"

SUPABASE_URL="${SUPABASE_URL:-https://itcarzyhlkiovskfdqut.supabase.co}"
SUPABASE_KEY="${SUPABASE_KEY:-sb_publishable_IG4iiZuIDAoxhzZSPZnRRA_xSl0VRfN}"

headers=(
  -H "apikey: $SUPABASE_KEY"
  -H "Authorization: Bearer $SUPABASE_KEY"
)

fetch_pages() {
  local base_url="$1"
  local output="$2"
  local page_dir
  page_dir="$(mktemp -d)"
  local offset=0
  local page=0
  local count=0
  while true; do
    local page_file="$page_dir/page_${page}.json"
    curl -sS --fail-with-body --max-time 60 --retry 18 --retry-all-errors --retry-delay 10 "${headers[@]}" \
      "${base_url}&limit=1000&offset=${offset}" \
      -o "$page_file"
    count="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$page_file")"
    if [[ "$count" -lt 1000 ]]; then
      break
    fi
    offset=$((offset + 1000))
    page=$((page + 1))
  done
  python3 - "$page_dir" "$output" <<'PY'
import json
import sys
from pathlib import Path

page_dir = Path(sys.argv[1])
output = Path(sys.argv[2])
rows = []
for path in sorted(page_dir.glob("page_*.json"), key=lambda item: int(item.stem.split("_")[1])):
    rows.extend(json.loads(path.read_text(encoding="utf-8")))
output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  rm -rf "$page_dir"
}

# Blind-safe song metadata. No lyrics, VA judgments, reviews, or section labels.
fetch_pages \
  "$SUPABASE_URL/rest/v1/song_review_summary?select=id,title,artist,audio_path,duration_sec,source,status,human_annotation_count,is_reviewed&status=eq.reviewed&order=id.asc" \
  "$OUT_DIR/reviewed_songs_allowed.json"

# Boundaries are used only after frame-level VA prediction for aggregation.
# section_type, confidence, qc_status, review history, and lyrics are excluded.
fetch_pages \
  "$SUPABASE_URL/rest/v1/human_annotations_readable?select=human_annotation_id,song_id,section_order,start_time_sec,end_time_sec,duration_sec,song_status&song_status=eq.reviewed&order=song_id.asc,section_order.asc,human_annotation_id.asc" \
  "$OUT_DIR/section_boundaries_allowed.json"

python3 - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
songs = json.loads((root / "reviewed_songs_allowed.json").read_text(encoding="utf-8"))
sections = json.loads((root / "section_boundaries_allowed.json").read_text(encoding="utf-8"))
print(f"reviewed_songs={len(songs)}")
print(f"section_boundary_rows={len(sections)}")
print(f"output_dir={root}")
PY
