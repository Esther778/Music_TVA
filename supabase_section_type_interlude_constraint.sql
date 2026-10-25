-- Run this in Supabase SQL Editor.
-- Purpose: allow the new `interlude` section label in existing section tables.
-- This changes constraints only. It does not create tables or views.

alter table public.human_annotations
  drop constraint if exists human_annotations_section_type_check;

alter table public.human_annotations
  add constraint human_annotations_section_type_check
  check (
    section_type in (
      'intro',
      'verse',
      'pre_chorus',
      'chorus',
      'bridge',
      'interlude',
      'outro'
    )
  );

alter table public.auto_sections
  drop constraint if exists auto_sections_section_type_check;

alter table public.auto_sections
  add constraint auto_sections_section_type_check
  check (
    section_type in (
      'intro',
      'verse',
      'pre_chorus',
      'chorus',
      'bridge',
      'interlude',
      'outro'
    )
  );
