#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/data/manifests/raw"
mkdir -p "$OUT_DIR"

SUPABASE_URL="${SUPABASE_URL:-https://itcarzyhlkiovskfdqut.supabase.co}"
SUPABASE_KEY="${SUPABASE_KEY:-sb_publishable_IG4iiZuIDAoxhzZSPZnRRA_xSl0VRfN}"
DATE_TAG="${1:-20260613}"

headers=(
  -H "apikey: $SUPABASE_KEY"
  -H "Authorization: Bearer $SUPABASE_KEY"
)

curl -sS --max-time 60 "${headers[@]}" \
  "$SUPABASE_URL/rest/v1/song_review_summary?select=id,title,artist,audio_path,duration_sec,duration_label,source,status,created_at,updated_at,latest_run_id,auto_section_count,human_annotation_count,is_reviewed&status=eq.reviewed&order=updated_at.asc,id.asc" \
  -o "$OUT_DIR/reviewed_songs_${DATE_TAG}.json"

curl -sS --max-time 60 "${headers[@]}" \
  "$SUPABASE_URL/rest/v1/human_annotations_readable?select=song_id,song_title,song_artist,section_order,section_type,start_time_sec,end_time_sec,duration_sec,song_status,is_approved,confidence,human_annotation_id,updated_at&song_status=eq.reviewed&order=song_id.asc,section_order.asc" \
  -o "$OUT_DIR/human_annotations_${DATE_TAG}.json"

echo "$OUT_DIR/reviewed_songs_${DATE_TAG}.json"
echo "$OUT_DIR/human_annotations_${DATE_TAG}.json"
