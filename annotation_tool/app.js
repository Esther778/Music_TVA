const SUPABASE_URL = "https://itcarzyhlkiovskfdqut.supabase.co";
const SUPABASE_KEY = "sb_publishable_IG4iiZuIDAoxhzZSPZnRRA_xSl0VRfN";

const client = window.supabase.createClient(SUPABASE_URL, SUPABASE_KEY);

const state = {
  songs: [],
  selectedSong: null,
  run: null,
  sections: [],
  annotations: [],
  selectedSection: null,
};

const els = {
  refreshButton: document.getElementById("refreshButton"),
  songCount: document.getElementById("songCount"),
  reviewCount: document.getElementById("reviewCount"),
  songSearch: document.getElementById("songSearch"),
  songList: document.getElementById("songList"),
  songTitle: document.getElementById("songTitle"),
  songMeta: document.getElementById("songMeta"),
  runMeta: document.getElementById("runMeta"),
  reviewFilter: document.getElementById("reviewFilter"),
  approveAllVisible: document.getElementById("approveAllVisible"),
  audioPlayer: document.getElementById("audioPlayer"),
  timeline: document.getElementById("timeline"),
  sectionList: document.getElementById("sectionList"),
  selectedBadge: document.getElementById("selectedBadge"),
  editForm: document.getElementById("editForm"),
  editType: document.getElementById("editType"),
  editStart: document.getElementById("editStart"),
  editEnd: document.getElementById("editEnd"),
  editConfidence: document.getElementById("editConfidence"),
  editComment: document.getElementById("editComment"),
  seekStart: document.getElementById("seekStart"),
  markUnsure: document.getElementById("markUnsure"),
  boundaryConfidence: document.getElementById("boundaryConfidence"),
  typeConfidence: document.getElementById("typeConfidence"),
  reviewFlag: document.getElementById("reviewFlag"),
  lyricEvidence: document.getElementById("lyricEvidence"),
  vocalEvidence: document.getElementById("vocalEvidence"),
  acousticEvidence: document.getElementById("acousticEvidence"),
  toast: document.getElementById("toast"),
};

const sectionTypes = ["intro", "verse", "pre_chorus", "chorus", "bridge", "outro"];

async function loadSongs() {
  const { data, error } = await client
    .from("songs")
    .select("*")
    .order("created_at", { ascending: false });
  if (error) return showToast(error.message, true);
  state.songs = data || [];
  renderSongs();
  els.songCount.textContent = state.songs.length;
}

function renderSongs() {
  const query = els.songSearch.value.trim().toLowerCase();
  const songs = state.songs.filter((song) => {
    const haystack = `${song.id} ${song.title} ${song.artist || ""}`.toLowerCase();
    return haystack.includes(query);
  });
  els.songList.innerHTML = "";
  for (const song of songs) {
    const card = document.createElement("div");
    card.className = `song-card ${state.selectedSong?.id === song.id ? "active" : ""}`;
    card.innerHTML = `
      <strong>${escapeHtml(song.title)}</strong>
      <span>${escapeHtml(song.artist || "unknown")} · ${escapeHtml(song.status || "pending")}</span>
      <span>${escapeHtml(song.id)}</span>
    `;
    card.addEventListener("click", () => selectSong(song.id));
    els.songList.appendChild(card);
  }
}

async function selectSong(songId) {
  const song = state.songs.find((item) => item.id === songId);
  state.selectedSong = song;
  state.selectedSection = null;
  renderSongs();
  els.songTitle.textContent = song.title;
  els.songMeta.textContent = `${song.artist || "unknown"} · ${formatTime(song.duration_sec || 0)} · ${song.id}`;
  els.audioPlayer.src = song.audio_path || "";

  const { data: runs, error: runError } = await client
    .from("model_runs")
    .select("*")
    .eq("song_id", songId)
    .order("created_at", { ascending: false })
    .limit(1);
  if (runError) return showToast(runError.message, true);
  state.run = runs?.[0] || null;

  if (!state.run) {
    state.sections = [];
    state.annotations = [];
    renderSections();
    return showToast("这首歌还没有 model_run");
  }

  const [{ data: sections, error: sectionError }, { data: annotations, error: annotationError }] = await Promise.all([
    client
      .from("auto_sections")
      .select("*")
      .eq("run_id", state.run.id)
      .order("section_index", { ascending: true }),
    client
      .from("human_annotations")
      .select("*")
      .eq("song_id", songId)
      .order("created_at", { ascending: false }),
  ]);
  if (sectionError) return showToast(sectionError.message, true);
  if (annotationError) return showToast(annotationError.message, true);
  state.sections = sections || [];
  state.annotations = annotations || [];
  renderSections();
}

function latestAnnotation(section) {
  return state.annotations.find((item) => item.source_auto_section_id === section.id);
}

function visibleSections() {
  const filter = els.reviewFilter.value;
  return state.sections.filter((section) => {
    const approved = Boolean(latestAnnotation(section));
    if (filter === "review") return section.need_human_review || Number(section.type_confidence) < 0.74 || Number(section.boundary_confidence) < 0.7;
    if (filter === "approved") return approved;
    return true;
  });
}

function renderSections() {
  const sections = visibleSections();
  els.reviewCount.textContent = state.sections.filter((section) => section.need_human_review).length;
  els.runMeta.textContent = state.run ? `${state.run.pipeline_version} · ${state.run.model_version}` : "";
  renderTimeline();
  els.sectionList.innerHTML = "";
  els.sectionList.classList.toggle("empty", sections.length === 0);
  if (!sections.length) {
    els.sectionList.textContent = state.run ? "当前过滤条件下没有 section" : "没有模型结果";
    clearInspector();
    return;
  }
  for (const section of sections) {
    const annotation = latestAnnotation(section);
    const row = document.createElement("div");
    row.className = `section-row ${state.selectedSection?.id === section.id ? "active" : ""}`;
    row.innerHTML = `
      <div class="section-time">${formatTime(section.start_time_sec)}<br>${formatTime(section.end_time_sec)}</div>
      <div class="section-main">
        <strong style="color:${typeColor(section.section_type)}">${section.section_type}</strong>
        <p>${shortEvidence(section)}</p>
      </div>
      <div class="risk ${section.need_human_review ? "review" : ""}">${annotation ? "saved" : riskLabel(section)}</div>
    `;
    row.addEventListener("click", () => selectSection(section.id));
    els.sectionList.appendChild(row);
  }
  if (state.selectedSection) fillInspector(state.selectedSection);
}

function renderTimeline() {
  els.timeline.innerHTML = "";
  if (!state.sections.length) return;
  const total = Math.max(...state.sections.map((section) => Number(section.end_time_sec)));
  for (const section of state.sections) {
    const seg = document.createElement("div");
    seg.className = "timeline-segment";
    seg.dataset.type = section.section_type;
    seg.style.width = `${((Number(section.end_time_sec) - Number(section.start_time_sec)) / total) * 100}%`;
    seg.textContent = section.section_type.replace("_", " ");
    seg.title = `${formatTime(section.start_time_sec)}-${formatTime(section.end_time_sec)} ${section.section_type}`;
    seg.addEventListener("click", () => selectSection(section.id));
    els.timeline.appendChild(seg);
  }
}

function selectSection(sectionId) {
  const section = state.sections.find((item) => item.id === sectionId);
  state.selectedSection = section;
  fillInspector(section);
  renderSections();
}

function fillInspector(section) {
  const annotation = latestAnnotation(section);
  els.selectedBadge.textContent = `#${section.section_index}`;
  els.editType.value = annotation?.section_type || section.section_type;
  els.editStart.value = Number(annotation?.start_time_sec ?? section.start_time_sec).toFixed(2);
  els.editEnd.value = Number(annotation?.end_time_sec ?? section.end_time_sec).toFixed(2);
  els.editConfidence.value = annotation?.confidence ?? 1;
  els.editComment.value = annotation?.comment || "";
  els.boundaryConfidence.textContent = numberText(section.boundary_confidence);
  els.typeConfidence.textContent = numberText(section.type_confidence);
  els.reviewFlag.textContent = section.need_human_review ? "true" : "false";
  els.lyricEvidence.textContent = section.lyric_evidence || "-";
  els.vocalEvidence.textContent = section.vocal_evidence || "-";
  els.acousticEvidence.textContent = section.acoustic_evidence || "-";
}

function clearInspector() {
  state.selectedSection = null;
  els.selectedBadge.textContent = "未选择";
  els.editForm.reset();
  els.boundaryConfidence.textContent = "-";
  els.typeConfidence.textContent = "-";
  els.reviewFlag.textContent = "-";
  els.lyricEvidence.textContent = "-";
  els.vocalEvidence.textContent = "-";
  els.acousticEvidence.textContent = "-";
}

async function saveAnnotation(action = "edit") {
  const section = state.selectedSection;
  if (!section || !state.selectedSong || !state.run) return showToast("先选择一个 section", true);
  const payload = {
    song_id: state.selectedSong.id,
    source_run_id: state.run.id,
    source_auto_section_id: section.id,
    annotator: "local_reviewer",
    section_index: section.section_index,
    start_time_sec: Number(els.editStart.value),
    end_time_sec: Number(els.editEnd.value),
    section_type: els.editType.value,
    confidence: Number(els.editConfidence.value),
    is_approved: action !== "unsure",
    comment: els.editComment.value,
  };
  const { data, error } = await client.from("human_annotations").insert(payload).select().single();
  if (error) return showToast(error.message, true);
  const reviewPayload = {
    song_id: state.selectedSong.id,
    run_id: state.run.id,
    auto_section_id: section.id,
    human_annotation_id: data.id,
    action,
    reviewer: "local_reviewer",
    before_json: section,
    after_json: payload,
  };
  const { error: reviewError } = await client.from("section_reviews").insert(reviewPayload);
  if (reviewError) return showToast(reviewError.message, true);
  state.annotations.unshift(data);
  await client.from("songs").update({ status: "in_review" }).eq("id", state.selectedSong.id);
  showToast(action === "approve" ? "已批准" : "已保存");
  renderSections();
}

async function approveAllVisible() {
  const sections = visibleSections();
  for (const section of sections) {
    if (latestAnnotation(section)) continue;
    state.selectedSection = section;
    els.editType.value = section.section_type;
    els.editStart.value = Number(section.start_time_sec).toFixed(2);
    els.editEnd.value = Number(section.end_time_sec).toFixed(2);
    els.editConfidence.value = 1;
    els.editComment.value = "approved as model output";
    await saveAnnotation("approve");
  }
}

function riskLabel(section) {
  if (section.need_human_review) return "review";
  if (Number(section.type_confidence) < 0.74) return "type";
  if (Number(section.boundary_confidence) < 0.7) return "edge";
  return "ok";
}

function shortEvidence(section) {
  const text = section.lyric_evidence || section.acoustic_evidence || "";
  return text.length > 100 ? `${text.slice(0, 100)}...` : text;
}

function typeColor(type) {
  return getComputedStyle(document.documentElement).getPropertyValue(`--${type}`) || "#17202a";
}

function formatTime(value) {
  const num = Number(value || 0);
  const minutes = Math.floor(num / 60);
  const seconds = (num - minutes * 60).toFixed(2).padStart(5, "0");
  return `${minutes}:${seconds}`;
}

function numberText(value) {
  if (value === null || value === undefined) return "-";
  return Number(value).toFixed(2);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function showToast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.style.background = isError ? "#bb2b2b" : "#17202a";
  els.toast.classList.add("visible");
  setTimeout(() => els.toast.classList.remove("visible"), 2200);
}

els.refreshButton.addEventListener("click", loadSongs);
els.songSearch.addEventListener("input", renderSongs);
els.reviewFilter.addEventListener("change", renderSections);
els.approveAllVisible.addEventListener("click", approveAllVisible);
els.seekStart.addEventListener("click", () => {
  if (!state.selectedSection) return;
  els.audioPlayer.currentTime = Number(state.selectedSection.start_time_sec);
  els.audioPlayer.play();
});
els.markUnsure.addEventListener("click", () => saveAnnotation("unsure"));
els.editForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const section = state.selectedSection;
  if (!section) return;
  const unchanged =
    els.editType.value === section.section_type &&
    Math.abs(Number(els.editStart.value) - Number(section.start_time_sec)) < 0.01 &&
    Math.abs(Number(els.editEnd.value) - Number(section.end_time_sec)) < 0.01;
  saveAnnotation(unchanged ? "approve" : "edit");
});

loadSongs();
