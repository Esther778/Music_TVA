-- Run this in the Supabase SQL Editor.
-- Purpose:
--   Create a new, clearly separated table for the authoritative paper dataset
--   based on the QQ Music top-60 Huayu/Chinese-artist sampling rule.
--   English stage names are included when the artist is a Chinese-language
--   singer/group, for example BY2 and S.H.E.
--
-- This does not modify existing QC tables:
--   public.songs
--   public.human_annotations
--   public.section_reviews
--   public.human_annotations_readable

create table if not exists public.paper_authoritative_top60_song_pool (
  id bigserial primary key,
  dataset_version text not null default 'qq_music_top60_v1',
  dataset_role text not null default 'new_authoritative_100',

  qq_rank integer not null check (qq_rank between 1 and 60),
  qq_artist_name text not null,
  rank_band text not null check (rank_band in ('rank_01_10', 'rank_11_30', 'rank_31_60')),
  target_song_count integer not null check (target_song_count in (1, 2, 3)),
  quota_slot integer not null check (quota_slot >= 1 and quota_slot <= target_song_count),

  song_id text references public.songs(id) on update cascade on delete set null,
  song_title text,
  song_artist text,
  local_folder text check (local_folder in ('songs', 'songs2')),
  source_status text not null default 'pending_selection'
    check (source_status in ('retained_from_old97', 'new_candidate', 'pending_selection', 'excluded')),

  ranking_source text not null default 'QQ Music Singer Chart',
  artist_population_rule text not null default 'QQ Music chart Huayu/Chinese-language artists, including Chinese/Taiwan/Hong Kong/Mainland singers and Chinese-language groups even when the stage name is written in English, such as BY2 and S.H.E.',
  ranking_snapshot_date date,
  selection_rule text not null default 'Ranks 1-10: 3 songs per artist; ranks 11-30: 2 songs per artist; ranks 31-60: 1 song per artist.',
  evidence_url text,
  notes text,

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  unique (dataset_version, qq_rank, quota_slot)
);

comment on table public.paper_authoritative_top60_song_pool is
  'Authoritative paper dataset song pool. Separate from QC source tables; links to public.songs by song_id when a selected song exists in Supabase.';

comment on column public.paper_authoritative_top60_song_pool.dataset_version is
  'Dataset version label, e.g. qq_music_top60_v1.';

comment on column public.paper_authoritative_top60_song_pool.dataset_role is
  'Keeps this new authoritative 100-song dataset distinct from the old 97-song exploratory corpus.';

comment on column public.paper_authoritative_top60_song_pool.song_id is
  'Nullable reference to public.songs.id. Leave null for future songs not yet imported to Supabase.';

comment on column public.paper_authoritative_top60_song_pool.source_status is
  'retained_from_old97 for songs kept from the old 97-song analysis; new_candidate for future added songs; pending_selection before final choice; excluded if intentionally not used.';

alter table public.paper_authoritative_top60_song_pool
  add column if not exists artist_population_rule text not null
  default 'QQ Music chart Huayu/Chinese-language artists, including Chinese/Taiwan/Hong Kong/Mainland singers and Chinese-language groups even when the stage name is written in English, such as BY2 and S.H.E.';

comment on column public.paper_authoritative_top60_song_pool.artist_population_rule is
  'Defines the ranked artist population. English stage names are included when the artist is a Huayu/Chinese-language artist, for example BY2 and S.H.E.';

create index if not exists paper_authoritative_top60_song_pool_song_id_idx
  on public.paper_authoritative_top60_song_pool(song_id);

create index if not exists paper_authoritative_top60_song_pool_rank_idx
  on public.paper_authoritative_top60_song_pool(dataset_version, qq_rank, quota_slot);

create or replace function public.set_paper_authoritative_top60_song_pool_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists set_paper_authoritative_top60_song_pool_updated_at
  on public.paper_authoritative_top60_song_pool;

create trigger set_paper_authoritative_top60_song_pool_updated_at
before update on public.paper_authoritative_top60_song_pool
for each row
execute function public.set_paper_authoritative_top60_song_pool_updated_at();

grant select, insert, update, delete on public.paper_authoritative_top60_song_pool to anon, authenticated;
grant usage, select on sequence public.paper_authoritative_top60_song_pool_id_seq to anon, authenticated;

-- Write smoke test. This inserts and immediately removes one row.
insert into public.paper_authoritative_top60_song_pool (
  dataset_version,
  qq_rank,
  qq_artist_name,
  rank_band,
  target_song_count,
  quota_slot,
  source_status,
  notes
) values (
  '__write_test__',
  1,
  'WRITE_TEST',
  'rank_01_10',
  3,
  1,
  'pending_selection',
  'Temporary write test row created by setup SQL.'
)
on conflict (dataset_version, qq_rank, quota_slot) do update
set notes = excluded.notes;

delete from public.paper_authoritative_top60_song_pool
where dataset_version = '__write_test__';
