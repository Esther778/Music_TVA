-- Roll database review state back to before 2026-06-09 20:00 Asia/Shanghai.
-- Equivalent UTC cutoff: 2026-06-09 12:00:00+00.
--
-- Run this in Supabase SQL Editor only after reviewing the preview SELECT output.
-- It backs up affected rows before changing anything.

begin;

-- 1) Backup rows that will be touched. These backup tables are kept in public.
create table if not exists public.rollback_backup_20260609_2000_human_annotations as
select distinct h.*
from public.human_annotations h
where h.updated_at >= timestamptz '2026-06-09 12:00:00+00'
   or h.created_at >= timestamptz '2026-06-09 12:00:00+00'
   or h.id in (
     select distinct human_annotation_id
     from public.section_reviews
     where created_at >= timestamptz '2026-06-09 12:00:00+00'
       and human_annotation_id is not null
   );

create table if not exists public.rollback_backup_20260609_2000_section_reviews as
select *
from public.section_reviews
where created_at >= timestamptz '2026-06-09 12:00:00+00';

-- 2) Build a temp restore set from the first post-cutoff review whose before_json
-- is actually the previous human_annotations row.
create temp table rollback_restore_human_annotations as
select distinct on (r.human_annotation_id)
  r.human_annotation_id as id,
  r.before_json as row_json,
  r.created_at as first_bad_review_at
from public.section_reviews r
where r.created_at >= timestamptz '2026-06-09 12:00:00+00'
  and r.human_annotation_id is not null
  and r.before_json ? 'id'
  and r.before_json->>'id' = r.human_annotation_id::text
order by r.human_annotation_id, r.created_at asc;

-- 3) Preview counts inside the transaction.
select
  (select count(*) from rollback_restore_human_annotations) as human_annotations_to_restore,
  (select count(*) from public.human_annotations where created_at >= timestamptz '2026-06-09 12:00:00+00') as new_human_annotations_to_delete,
  (select count(*) from public.section_reviews where created_at >= timestamptz '2026-06-09 12:00:00+00') as section_reviews_to_delete;

-- 4) Restore changed pre-existing human_annotations rows.
update public.human_annotations h
set
  song_id = rr.row_json->>'song_id',
  source_run_id = nullif(rr.row_json->>'source_run_id', '')::uuid,
  source_auto_section_id = nullif(rr.row_json->>'source_auto_section_id', '')::uuid,
  annotator = rr.row_json->>'annotator',
  section_index = nullif(rr.row_json->>'section_index', '')::integer,
  start_time_sec = nullif(rr.row_json->>'start_time_sec', '')::double precision,
  end_time_sec = nullif(rr.row_json->>'end_time_sec', '')::double precision,
  section_type = rr.row_json->>'section_type',
  confidence = nullif(rr.row_json->>'confidence', '')::double precision,
  is_approved = nullif(rr.row_json->>'is_approved', '')::boolean,
  comment = rr.row_json->>'comment',
  created_at = nullif(rr.row_json->>'created_at', '')::timestamptz,
  updated_at = nullif(rr.row_json->>'updated_at', '')::timestamptz
from rollback_restore_human_annotations rr
where h.id = rr.id;

-- 5) Delete review activity after the cutoff, so readable activity views also go
-- back to pre-20:00 state. Delete these before deleting new human annotations in
-- case section_reviews has a foreign key to human_annotations.
delete from public.section_reviews r
where r.created_at >= timestamptz '2026-06-09 12:00:00+00';

-- 6) Delete human_annotations created after the cutoff because they did not exist
-- in the requested pre-20:00 state.
delete from public.human_annotations h
where h.created_at >= timestamptz '2026-06-09 12:00:00+00';

-- 7) Post-check. These should be zero for post-cutoff review rows.
select
  (select count(*) from public.human_annotations where updated_at >= timestamptz '2026-06-09 12:00:00+00' or created_at >= timestamptz '2026-06-09 12:00:00+00') as remaining_post_cutoff_human_annotations,
  (select count(*) from public.section_reviews where created_at >= timestamptz '2026-06-09 12:00:00+00') as remaining_post_cutoff_section_reviews;

commit;
