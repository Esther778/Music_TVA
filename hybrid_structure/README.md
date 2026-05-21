# Functional Pop-Song Section Pipeline

Current direction: this is not an energy-change detector and not a lyrics-only
splitter. The pipeline treats section segmentation as:

```text
MP3 -> sentence-level lyrics with sentence audio features
lyrics semantics create draft sections first
instrumental gaps are resolved by duration and acoustic similarity
acoustic features validate or override only when confidence is strong
```

## Recommended Command

```bash
.venv/bin/python hybrid_structure/agentic/orchestrator.py \
  --audio "songs/周深-生活总该迎着光亮.mp3" \
  --title "生活总该迎着光亮" \
  --song-id life_should_face_light \
  --work-dir outputs/functional_sections \
  --output outputs/functional_sections/life_should_face_light_agentic_sections.csv \
  --reuse-transcript \
  --vocals-stem "outputs/vocal_separation/htdemucs/周深-生活总该迎着光亮/vocals.wav" \
  --candidate-output outputs/functional_sections/life_should_face_light_boundary_candidates.csv
```

This is the current recommended entry point. It splits the work into auditable
modules: lyric source, lyric function, acoustic boundary candidates, vocal
activity, fusion, and optional evaluation.

For another song, change `--audio`, `--title`, `--song-id`, `--output`, and
optionally `--candidate-output`.

If the transcript already exists in `--work-dir`, add `--reuse-transcript` to
skip Whisper and only rerun segmentation.

Use `--lyrics-file path/to/song.lrc` or `--lyrics-csv path/to/transcript.csv`
when a searched or manually curated timed-lyrics source is available. The
current implementation stores only section evidence and timing outputs, not
full third-party lyrics.

The older direct script still works for compatibility:

```bash
.venv/bin/python hybrid_structure/functional_section_pipeline.py \
  --audio "songs/周深-生活总该迎着光亮.mp3" \
  --title "生活总该迎着光亮" \
  --song-id life_should_face_light \
  --model small \
  --language zh \
  --work-dir outputs/functional_sections \
  --output outputs/functional_sections/life_should_face_light_vocal_corrected_sections.csv \
  --reuse-transcript \
  --vocals-stem "outputs/vocal_separation/htdemucs/周深-生活总该迎着光亮/vocals.wav"
```

For better intro/bridge/outro detection, run Demucs first:

```bash
.venv/bin/python -m demucs --two-stems=vocals -n htdemucs \
  --out outputs/vocal_separation \
  "songs/周深-生活总该迎着光亮.mp3"
```

Then pass the generated `vocals.wav` through `--vocals-stem`. Long low-vocal
regions are forced out of sung sections as intro, bridge, or outro candidates.

## Output

The current output CSV fields are:

```text
song_id
section_index
start_time
end_time
duration
section_type
structure_label
lyric_evidence
vocal_evidence
acoustic_evidence
boundary_confidence
type_confidence
need_human_review
```

This format intentionally does not pretend the automatic result is final. It
exports a candidate section boundary, candidate type, evidence, confidence, and
review flag.

The agentic command also writes intermediate CSV files by default:

```text
<song_id>_sentence_segments.csv
<song_id>_lyric_draft_sections.csv
<song_id>_gap_resolution.csv
<song_id>_acoustic_validation.csv
```

These files are the main debugging surface. The intended order is:

```text
sentence_segments -> lyric_draft_sections -> gap_resolution -> acoustic_validation -> final sections
```

Current type labels also pass through a global type refinement step:

```text
type_refinement -> final sections
```

## Current Results

Latest generated files:

```text
outputs/functional_sections/life_should_face_light_agentic_sections.csv
outputs/functional_sections/anohana_agentic_sections.csv
outputs/functional_sections/time_flood_agentic_sections.csv
outputs/functional_sections/life_should_face_light_boundary_candidates.csv
outputs/functional_sections/anohana_boundary_candidates.csv
outputs/functional_sections/time_flood_boundary_candidates.csv
outputs/functional_sections/life_should_face_light_sentence_segments.csv
outputs/functional_sections/anohana_sentence_segments.csv
outputs/functional_sections/time_flood_sentence_segments.csv
outputs/functional_sections/anohana_type_refinement.csv
```

Supporting transcripts:

```text
outputs/functional_sections/周深-生活总该迎着光亮_whisper_segments.csv
outputs/functional_sections/周深-一期一会《未闻花名》_whisper_segments.csv
outputs/functional_sections/程响-时光洪流_whisper_segments.csv
```

## Core Logic

- Whisper or an external `.lrc` creates sentence-level vocal/lyric timestamps.
- Every lyric sentence gets paired with its corresponding audio span and
  sentence-level acoustic features: RMS, onset, centroid, novelty, and chroma
  summary.
- Lyric semantics and repetition create the draft sections before instrumental
  gaps or acoustic validation are applied.
- `intro` and `outro` are reserved for lyric-free or non-lexical vocal regions.
  Any section containing complete lyrical lines must be labeled by its sung
  function, such as verse, chorus, pre_chorus, or bridge.
- Long instrumental gaps between lyric sections become bridge candidates.
- Short instrumental gaps are absorbed into the previous or next section by
  acoustic similarity.
- Acoustic features validate draft boundaries. They only override the lyric
  draft when the acoustic confidence is strong enough to beat lyric confidence.
- Title hits and repeated hook material seed `chorus`, but final chorus lyric
  variation is kept as chorus when the hook function remains.
- `pre_chorus` is inferred as a contiguous buildup between verse-like material
  and the chorus hook using lyric gaps, phrasing density, and structural
  position, not a fixed number of lines.
- Mid-song low-vocal or one-off contrast passages after chorus cycles become
  `bridge` candidates when they form a sustained connecting/turning section.
- `bridge` can be instrumental or lyrical. For lyrical bridge candidates, the
  deciding evidence is not "has no vocals" but mid/late structural position,
  contrast against the surrounding chorus/verse cycle, acoustic profile change,
  and connecting/turning function.
- Repeated or highly similar lyric material increases the probability that two
  sections share the same functional type, but it is not a hard rule. Similar
  lyrics can still be labeled differently when position, neighboring sections,
  and acoustic role show different function, such as chorus hook versus
  post-chorus/bridge transition.
- `pre_chorus` must functionally lead into a chorus. A section previously
  drafted as pre_chorus is reclassified when it sits after a chorus, does not
  immediately resolve into a chorus, or behaves as a sustained transition.
- Continuous sung lyric paragraphs are protected from weak internal bridge
  splits. A bridge inside a paragraph must be sustained relative to the song's
  own pre-chorus scale, or have a real vocal/lyric break, otherwise it is kept
  with the surrounding chorus/verse function.
- Overlong sung sections are split at internal lyric-cycle restarts when the
  section exceeds the song's own expected chorus scale. This prevents a final
  double chorus from collapsing into one unrealistic section.
- Long lyric-free interior gaps become `bridge` candidates, because a long
  no-vocal span is a functional break rather than a normal sung section.
- Final short repeated tail lines with complete lyrics are kept as sung coda
  under the closest sung function, currently `chorus` in the six-label output.
  Only lyric-free or non-lexical final material is labeled `outro`.
- Acoustic features include RMS, onset strength, spectral centroid, chroma,
  MFCC, and a combined novelty curve.
- Optional Demucs vocal separation detects low-vocal regions and forces them out
  of adjacent sung sections as intro/bridge/outro candidates.
- Boundaries can snap to nearby novelty peaks, but weak acoustic novelty does
  not erase a strong lyric/vocal functional boundary.
- Pure acoustic changes without lyric/vocal or structural function are treated
  as candidates, not automatic sections.
- Low-confidence boundaries/types, overlong choruses, and suspicious short
  sections are marked `need_human_review=true`.

## Legacy Files

`auto_pipeline.py` and `segment_hybrid.py` are earlier prototypes. They are kept
for comparison/reference, but `functional_section_pipeline.py` is the current
entry point.
