#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import functional_section_pipeline as pipeline

from agentic.acoustic_boundary_agent import AcousticBoundaryAgent
from agentic.evaluator import evaluate_sections, parse_manual_csv
from agentic.lyric_function_agent import LyricFunctionAgent
from agentic.lyric_source_agent import LyricSourceAgent
from agentic.sentence_first_pipeline import (
    build_sentence_features,
    draft_sections_from_lyrics,
    finalize_sections,
    resolve_instrumental_gaps,
    validate_with_acoustics,
    write_acoustic_validation,
    write_draft_sections,
    write_gap_resolution,
    write_sentence_segments,
    write_sentence_segments_redacted,
)
from agentic.type_refiner import refine_section_types, write_type_decisions
from agentic.vocal_activity_agent import VocalActivityAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agentic functional section segmentation pipeline.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--song-id", default="")
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--work-dir", default="outputs/functional_sections")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-transcript", action="store_true")
    parser.add_argument("--track-name", help="Track name for synchronized lyric search.")
    parser.add_argument("--artist-name", help="Artist name for synchronized lyric search.")
    parser.add_argument("--no-lyric-search", action="store_true", help="Skip external synchronized lyric search.")
    parser.add_argument("--lyrics-csv")
    parser.add_argument("--lyrics-file", help="Optional timed lyrics file, currently .lrc or transcript-style .csv.")
    parser.add_argument("--lyrics-url", help="Optional webpage containing synchronized LRC lyrics.")
    parser.add_argument("--vocals-stem")
    parser.add_argument("--min-instrumental-gap", type=float, default=8.0)
    parser.add_argument("--candidate-output", help="Optional CSV for merged boundary candidates.")
    parser.add_argument("--sentence-output", help="Optional CSV for sentence-level lyric/audio features.")
    parser.add_argument("--draft-output", help="Optional CSV for lyric-only draft sections.")
    parser.add_argument("--gap-output", help="Optional CSV for instrumental gap decisions.")
    parser.add_argument("--validation-output", help="Optional CSV for acoustic validation decisions.")
    parser.add_argument("--type-output", help="Optional CSV for global section type refinement decisions.")
    parser.add_argument("--manual-csv", help="Optional manual annotation CSV for evaluation.")
    args = parser.parse_args()

    audio = Path(args.audio)
    work_dir = Path(args.work_dir)
    song_id = args.song_id or audio.stem

    lyric_agent = LyricSourceAgent()
    lines, lyric_source = lyric_agent.run(
        audio=audio,
        work_dir=work_dir,
        model=args.model,
        language=args.language,
        reuse_transcript=args.reuse_transcript,
        lyrics_csv=Path(args.lyrics_csv) if args.lyrics_csv else None,
        lyrics_file=Path(args.lyrics_file) if args.lyrics_file else None,
        lyrics_url=args.lyrics_url or "",
        track_name="" if args.no_lyric_search else (args.track_name or args.title),
        artist_name=args.artist_name or "",
        duration_seconds=None,
    )

    roles, lyric_function = LyricFunctionAgent().run(lines, args.title)
    features, acoustic = AcousticBoundaryAgent().run(
        audio,
        lines,
        Path(args.candidate_output) if args.candidate_output else None,
    )
    vocal = VocalActivityAgent().run(Path(args.vocals_stem) if args.vocals_stem else None)

    context = pipeline.structure_context(lines)
    sentence_rows = build_sentence_features(lines, roles, features)
    sentence_output = Path(args.sentence_output) if args.sentence_output else work_dir / f"{song_id}_sentence_segments.csv"
    if lyric_source.source in {"lrclib_synced_lyrics", "web_synced_lyrics", "local_timed_lyrics_file"}:
        write_sentence_segments_redacted(sentence_output, sentence_rows)
    else:
        write_sentence_segments(sentence_output, sentence_rows)

    draft_sections = draft_sections_from_lyrics(lines, roles, context)
    draft_output = Path(args.draft_output) if args.draft_output else work_dir / f"{song_id}_lyric_draft_sections.csv"
    write_draft_sections(draft_output, song_id, draft_sections)

    sections, gap_decisions = resolve_instrumental_gaps(draft_sections, features, float(features["duration"]), context)
    gap_output = Path(args.gap_output) if args.gap_output else work_dir / f"{song_id}_gap_resolution.csv"
    write_gap_resolution(gap_output, gap_decisions)

    if vocal.low_vocal_regions:
        sections = pipeline.apply_low_vocal_regions(sections, vocal.low_vocal_regions, float(features["duration"]))

    validations = validate_with_acoustics(sections, features)
    validation_output = Path(args.validation_output) if args.validation_output else work_dir / f"{song_id}_acoustic_validation.csv"
    write_acoustic_validation(validation_output, validations)

    sections = finalize_sections(sections, lines, features, args.title)
    sections, type_decisions = refine_section_types(sections, float(features["duration"]))
    type_output = Path(args.type_output) if args.type_output else work_dir / f"{song_id}_type_refinement.csv"
    write_type_decisions(type_output, type_decisions)
    pipeline.write_sections(Path(args.output), song_id, sections)

    print(f"Lyric source: {lyric_source.transcript_csv} ({lyric_source.source}, {lyric_source.line_count} lines)")
    print(f"Lyric function: {lyric_function.role_count} across {lyric_function.paragraph_count} paragraphs")
    print(f"Acoustic candidates: {len(acoustic.candidates)}")
    print(f"Vocal low regions: {len(vocal.low_vocal_regions)} from {vocal.source}")
    print(f"Sentence segments: {sentence_output}")
    print(f"Lyric draft: {draft_output}")
    print(f"Gap decisions: {gap_output}")
    print(f"Acoustic validation: {validation_output}")
    print(f"Type refinement: {type_output}")
    print(f"Sections: {args.output}")

    if args.manual_csv:
        report = evaluate_sections(sections, parse_manual_csv(Path(args.manual_csv)))
        print(
            "Evaluation: "
            f"label_accuracy={report.label_accuracy:.3f} "
            f"boundary_mae={report.boundary_mae:.2f}s "
            f"section_count={report.section_count}/{report.manual_count}"
        )

    for index, section in enumerate(sections, start=1):
        print(
            f"{index:02d} {pipeline.base.format_timestamp(section.start)}-{pipeline.base.format_timestamp(section.end)} "
            f"{pipeline.public_section_type(section.section_type)} "
            f"boundary={section.boundary_confidence:.2f} type={section.type_confidence:.2f} "
            f"review={str(section.need_human_review).lower()}"
        )


if __name__ == "__main__":
    main()
