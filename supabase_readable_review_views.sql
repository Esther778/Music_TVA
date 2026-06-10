-- Run this in Supabase SQL Editor.
-- Keep only one readable inspection view: public.human_annotations_readable.
-- This does not delete source tables. It only drops extra readable views and
-- recreates human_annotations_readable for all section-level human QC results.

drop view if exists public.review_activity_latest;
drop view if exists public.section_reviews_readable;
drop view if exists public.human_annotations_readable;

create or replace view public.human_annotations_readable
with (security_invoker = true)
as
with songs_base as (
  select
    s.id as song_id,
    s.title as song_title,
    s.artist as song_artist,
    s.status as song_status,
    s.updated_at as song_reviewed_at
  from public.songs s
),
latest_human_annotations as (
  select distinct on (
    h.song_id,
    coalesce(
      h.source_auto_section_id::text,
      h.section_index::text || ':' || h.start_time_sec::text || ':' || h.end_time_sec::text,
      h.id::text
    )
  )
    h.*
  from public.human_annotations h
  order by
    h.song_id,
    coalesce(
      h.source_auto_section_id::text,
      h.section_index::text || ':' || h.start_time_sec::text || ':' || h.end_time_sec::text,
      h.id::text
    ),
    h.updated_at desc nulls last,
    h.created_at desc nulls last,
    h.id desc
)
select
  rs.song_title,
  rs.song_artist,
  row_number() over (
    partition by rs.song_id
    order by
      h.start_time_sec nulls last,
      h.end_time_sec nulls last,
      h.section_index nulls last,
      h.id
  ) as section_order,
  h.section_type,
  h.start_time_sec,
  h.end_time_sec,
  round((h.end_time_sec - h.start_time_sec)::numeric, 2) as duration_sec,
  rs.song_reviewed_at,
  rs.song_status,
  rs.song_id,
  h.section_index as original_section_index,
  case
    when h.source_auto_section_id is null then 'human_added'
    when h.comment = '默认采用机器标注' then 'machine_confirmed'
    else 'human_modified'
  end as annotation_kind,
  h.is_approved,
  h.confidence,
  h.annotator,
  h.comment,
  h.source_run_id,
  h.source_auto_section_id,
  h.id as human_annotation_id,
  h.created_at,
  h.updated_at
from songs_base rs
join latest_human_annotations h on h.song_id = rs.song_id
order by
  case when rs.song_status = 'reviewed' then 0 else 1 end,
  rs.song_reviewed_at desc nulls last,
  rs.song_title,
  rs.song_artist,
  section_order;

grant select on public.human_annotations_readable to anon, authenticated;
