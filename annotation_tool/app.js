const SUPABASE_URL = "https://itcarzyhlkiovskfdqut.supabase.co";
const SUPABASE_KEY = "sb_publishable_IG4iiZuIDAoxhzZSPZnRRA_xSl0VRfN";

const client = window.supabase.createClient(SUPABASE_URL, SUPABASE_KEY);

window.__errors = [];
window.addEventListener("error", (event) => {
  window.__errors.push(event.message);
});
window.addEventListener("unhandledrejection", (event) => {
  window.__errors.push(event.reason?.message || String(event.reason));
});

const sectionTypes = ["intro", "verse", "pre_chorus", "chorus", "bridge", "interlude", "outro"];
const playbackRates = [1, 1.25, 1.5, 2, 0.75];
const AUTO_BOUNDARY_SYNC_COMMENT = "相邻 section 边界自动同步";
const DELETE_BOUNDARY_SYNC_COMMENT = "删除 section 后边界自动连接";
const SUBMIT_TIMEOUT_MS = 20000;
const TIME_EPSILON = 0.01;

const state = {
  songs: [],
  songOptionMap: new Map(),
  selectedSong: null,
  run: null,
  sections: [],
  annotations: [],
  reviews: [],
  lyrics: [],
  selectedSection: null,
  directoryOpen: true,
  deletedSectionIds: new Set(),
  deletedLocalAnnotationIds: new Set(),
  zoom: 1,
  currentLyricIndex: null,
  supportsQcStatus: true,
  previewEndTime: null,
  timelineDrag: null,
  isScrubbing: false,
  syncingSectionFromPlayback: false,
  submittingSectionIds: new Set(),
  confirmedSectionIds: new Set(),
  pendingSectionOrderChange: false,
  isSubmittingSong: false,
  manualWholeSongSubmitActive: false,
  toastTimer: null,
  selectionToken: 0,
  undoStack: [],
  lyricUndoStack: [],
  lastEditorSnapshotKey: "",
  lastLyricSnapshotKey: "",
  playbackRate: 1,
  draggedSectionId: null,
  pointerSectionDrag: null,
  lyricModalDrag: null,
  lyricModalResize: null,
  lyricModalFrame: null,
  sectionVisualEffects: new Map(),
};
window.qaState = state;

const els = {
  refreshButton: document.getElementById("refreshButton"),
  songCount: document.getElementById("songCount"),
  pendingCount: document.getElementById("pendingCount"),
  reviewedCount: document.getElementById("reviewedCount"),
  actionCount: document.getElementById("actionCount"),
  songPicker: document.getElementById("songPicker"),
  songOptions: document.getElementById("songOptions"),
  statusFilter: document.getElementById("statusFilter"),
  selectedSongCard: document.getElementById("selectedSongCard"),
  songDirectoryTitle: document.getElementById("songDirectoryTitle"),
  songDirectory: document.getElementById("songDirectory"),
  runMeta: document.getElementById("runMeta"),
  reviewFilter: document.getElementById("reviewFilter") || Object.assign(document.createElement("input"), { value: "all" }),
  finishReview: document.getElementById("finishReview"),
  audioPlayer: document.getElementById("audioPlayer"),
  playToggle: document.getElementById("playToggle"),
  currentTimeText: document.getElementById("currentTimeText"),
  progressRange: document.getElementById("progressRange"),
  durationText: document.getElementById("durationText"),
  speedToggle: document.getElementById("speedToggle"),
  timeline: document.getElementById("timeline"),
  zoomOut: document.getElementById("zoomOut"),
  zoomRange: document.getElementById("zoomRange"),
  zoomIn: document.getElementById("zoomIn"),
  lyricsViewport: document.getElementById("lyricsViewport"),
  lyricsExpand: document.getElementById("lyricsExpand"),
  lyricsModal: document.getElementById("lyricsModal"),
  lyricsModalPanel: document.querySelector(".lyrics-modal-panel"),
  lyricsModalClose: document.getElementById("lyricsModalClose"),
  lyricsModalPlay: document.getElementById("lyricsModalPlay"),
  lyricsModalResize: document.getElementById("lyricsModalResize"),
  lyricsBulkList: document.getElementById("lyricsBulkList"),
  lyricsBulkAdd: document.getElementById("lyricsBulkAdd"),
  lyricsBulkSave: document.getElementById("lyricsBulkSave"),
  sectionList: document.getElementById("sectionList"),
  selectedBadge: document.getElementById("selectedBadge"),
  finalResultSummary: document.getElementById("finalResultSummary"),
  editForm: document.getElementById("editForm"),
  typeQuickButtons: document.getElementById("typeQuickButtons"),
  editType: document.getElementById("editType"),
  editStart: document.getElementById("editStart"),
  editEnd: document.getElementById("editEnd"),
  prevSection: document.getElementById("prevSection"),
  seekStart: document.getElementById("seekStart"),
  useModel: document.getElementById("useModel"),
  addSection: document.getElementById("addSection"),
  deleteSection: document.getElementById("deleteSection"),
  resetCurrentSection: document.getElementById("resetCurrentSection"),
  resetSong: document.getElementById("resetSong"),
  markPendingReview: document.getElementById("markPendingReview"),
  nextSection: document.getElementById("nextSection"),
  saveEdit: document.getElementById("saveEdit"),
  bottomPlayer: document.getElementById("bottomPlayer"),
  bottomPlayToggle: document.getElementById("bottomPlayToggle"),
  bottomCurrentTimeText: document.getElementById("bottomCurrentTimeText"),
  bottomProgressRange: document.getElementById("bottomProgressRange"),
  bottomDurationText: document.getElementById("bottomDurationText"),
  bottomSpeedToggle: document.getElementById("bottomSpeedToggle"),
  boundaryConfidence: document.getElementById("boundaryConfidence"),
  typeConfidence: document.getElementById("typeConfidence"),
  reviewFlag: document.getElementById("reviewFlag"),
  lyricEvidence: document.getElementById("lyricEvidence"),
  vocalEvidence: document.getElementById("vocalEvidence"),
  acousticEvidence: document.getElementById("acousticEvidence"),
  toast: document.getElementById("toast"),
};

function renderTypeButtons() {
  els.typeQuickButtons.innerHTML = "";
  for (const type of sectionTypes) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = type.replace("_", " ");
    button.dataset.type = type;
    button.addEventListener("click", () => {
      rememberEditorState("修改 section type");
      els.editType.value = type;
      renderTypeButtonState();
    });
    els.typeQuickButtons.appendChild(button);
  }
}

async function loadSongs() {
  const [
    { data, error },
    { data: runs, error: runError },
    { data: songStatuses, error: statusError },
    { data: readableRows, error: readableError },
    { data: deletedRows, error: deletedError },
  ] = await Promise.all([
    client
      .from("song_review_summary")
      .select("*")
      .order("is_reviewed", { ascending: true })
      .order("title", { ascending: true }),
    client.from("model_runs").select("song_id"),
    client.from("songs").select("id,status"),
    client.from("human_annotations_readable").select("song_id,song_status,human_annotation_id"),
    client
      .from("section_reviews")
      .select("song_id,auto_section_id,action,after_json")
      .contains("after_json", { deleted: true }),
  ]);
  if (error) throw error;
  if (runError) throw runError;
  if (statusError) throw statusError;
  if (readableError) throw readableError;
  if (deletedError) throw deletedError;
  const readySongIds = new Set((runs || []).map((run) => run.song_id));
  const statusBySongId = new Map((songStatuses || []).map((song) => [song.id, song.status]));
  const readableSectionCountBySongId = new Map();
  const submittedReadableSectionCountBySongId = new Map();
  const deletedSectionCountBySongId = new Map();
  for (const row of readableRows || []) {
    if (!row.song_id || !row.human_annotation_id) continue;
    readableSectionCountBySongId.set(row.song_id, (readableSectionCountBySongId.get(row.song_id) || 0) + 1);
    submittedReadableSectionCountBySongId.set(row.song_id, (submittedReadableSectionCountBySongId.get(row.song_id) || 0) + 1);
  }
  const deletedSectionIdsBySongId = new Map();
  for (const row of deletedRows || []) {
    if (!row.song_id || !row.auto_section_id) continue;
    if (reviewAfterJson(row)?.deleted !== true) continue;
    if (!deletedSectionIdsBySongId.has(row.song_id)) deletedSectionIdsBySongId.set(row.song_id, new Set());
    deletedSectionIdsBySongId.get(row.song_id).add(row.auto_section_id);
  }
  for (const [songId, sectionIds] of deletedSectionIdsBySongId.entries()) {
    deletedSectionCountBySongId.set(songId, sectionIds.size);
  }
  state.songs = (data || [])
    .filter((song) => readySongIds.has(song.id))
    .map((song) => ({
      ...song,
      status: statusBySongId.get(song.id) || song.status,
      readable_section_count: readableSectionCountBySongId.get(song.id) || 0,
      submitted_readable_section_count: submittedReadableSectionCountBySongId.get(song.id) || 0,
      deleted_section_count: deletedSectionCountBySongId.get(song.id) || 0,
    }));
  if (state.selectedSong) {
    const updated = state.songs.find((song) => song.id === state.selectedSong.id);
    if (updated) state.selectedSong = updated;
    else state.selectedSong = null;
  }
  renderSongPicker();
  renderSummary();
  if (!state.selectedSong && state.songs.length === 1) {
    await selectSong(state.songs[0].id);
  }
}

async function refreshReviewData() {
  if (state.isSubmittingSong || state.submittingSectionIds.size) return showToast("正在提交中，不能刷新或切歌", true);
  if (els.refreshButton.disabled) return;
  const songId = state.selectedSong?.id;
  els.refreshButton.disabled = true;
  els.refreshButton.classList.add("refreshing");
  showToast("刷新中", false, { persistent: true });
  try {
    await loadSongs();
    if (songId) await selectSong(songId);
    showToast("刷新成功");
  } catch (error) {
    showToast(submitErrorMessage(error, "刷新失败"), true);
  } finally {
    els.refreshButton.disabled = false;
    els.refreshButton.classList.remove("refreshing");
  }
}

function renderSummary() {
  const reviewed = state.songs.filter((song) => isSongReviewedFromDatabase(song)).length;
  const currentSongCounts = state.selectedSong ? selectedSongCounts(state.selectedSong) : { reviewedSections: 0, total: 0 };
  els.songCount.textContent = state.songs.length;
  els.reviewedCount.textContent = reviewed;
  els.pendingCount.textContent = Math.max(0, state.songs.length - reviewed);
  els.actionCount.textContent = `${currentSongCounts.reviewedSections}/${currentSongCounts.total}`;
}

function renderSongPicker() {
  const filter = els.statusFilter.value;
  syncSummaryFilterState(filter);
  els.songDirectoryTitle.textContent = directoryTitle(filter);
  const songs = filteredSongs(filter, els.songPicker.value);
  state.songOptionMap.clear();
  els.songOptions.innerHTML = "";
  for (const song of songs) {
    const status = isSongReviewedFromDatabase(song) ? "已质检" : "未质检";
    const label = `${status} | ${song.title} | ${song.artist || "unknown"} | ${song.id}`;
    state.songOptionMap.set(label, song.id);
    const option = document.createElement("option");
    option.value = label;
    els.songOptions.appendChild(option);
  }
  renderSongDirectory(songs);
  renderSelectedSongCard();
}

function directoryTitle(filter) {
  return {
    all: "全部歌曲目录",
    pending: "未质检歌曲目录",
    reviewed: "已质检歌曲目录",
  }[filter] || "歌曲目录";
}

function filteredSongs(filter, query) {
  const text = String(query || "").trim();
  return state.songs.filter((song) => {
    const reviewed = isSongReviewedFromDatabase(song);
    if (filter === "reviewed" && !reviewed) return false;
    if (filter === "pending" && reviewed) return false;
    return fuzzySongMatch(song, text);
  });
}

function fuzzySongMatch(song, query) {
  if (!query) return true;
  const haystack = `${song.title} ${song.artist || ""} ${song.id}`.toLowerCase();
  const needle = query.toLowerCase().replace(/\s+/g, "");
  if (haystack.includes(query.toLowerCase())) return true;
  let pos = 0;
  for (const char of needle) {
    pos = haystack.indexOf(char, pos);
    if (pos === -1) return false;
    pos += 1;
  }
  return true;
}

function renderSongDirectory(songs) {
  els.songDirectory.classList.toggle("collapsed", !state.directoryOpen);
  if (!state.directoryOpen) return;
  if (!songs.length) {
    els.songDirectory.innerHTML = `<div class="directory-empty">没有匹配歌曲</div>`;
    return;
  }
  els.songDirectory.innerHTML = "";
  for (const song of songs) {
    const counts = state.selectedSong?.id === song.id ? selectedSongCounts(song) : databaseSongCounts(song);
    const reviewed = isSongReviewedFromDatabase(song);
    const item = document.createElement("button");
    item.type = "button";
    item.className = `directory-item ${state.selectedSong?.id === song.id ? "active" : ""}`;
    item.innerHTML = `
      <strong>${escapeHtml(song.title)}</strong>
      <span>${escapeHtml(song.artist || "unknown")} · ${reviewed ? "已质检" : "未质检"} · ${counts.reviewedSections}/${counts.total} 段</span>
    `;
    item.addEventListener("click", () => selectSong(song.id));
    els.songDirectory.appendChild(item);
  }
}

function syncSummaryFilterState(filter) {
  for (const card of document.querySelectorAll("[data-song-filter]")) {
    card.classList.toggle("active", card.dataset.songFilter === filter);
  }
}

function setSongFilter(filter) {
  els.statusFilter.value = filter;
  els.songPicker.value = "";
  state.directoryOpen = true;
  renderSongPicker();
  const firstMatch = filteredSongs(filter, "")[0];
  if (firstMatch && state.selectedSong?.id !== firstMatch.id) {
    selectSong(firstMatch.id);
  }
  els.songPicker.focus();
  els.songPicker.scrollIntoView({ block: "nearest", inline: "nearest" });
}

window.setSongFilterFromCard = setSongFilter;

function renderSelectedSongCard() {
  if (!state.selectedSong) {
    els.selectedSongCard.textContent = "未选择歌曲";
    return;
  }
  const song = state.selectedSong;
  const counts = selectedSongCounts(song);
  const reviewed = isSongReviewed(song);
  els.selectedSongCard.innerHTML = `
    <strong>${escapeHtml(song.title)}</strong>
    <span>${escapeHtml(song.artist || "unknown")} · ${escapeHtml(formatTime(song.duration_sec))}</span>
    <span class="${reviewed ? "status-reviewed" : "status-pending"}">${reviewed ? "已质检" : "未质检"} · ${counts.reviewedSections}/${counts.total} 段</span>
  `;
}

function selectedSongCounts(song) {
  if (state.selectedSong?.id === song.id && state.sections.length) {
    const sections = activeSections();
    const total = sections.length;
    const reviewedSections = sections.filter((section) => isSectionSubmitted(section)).length;
    const actions = manualReviewCount();
    return { total, actions, reviewedSections, isReviewed: isSongReviewedFromDatabase(song) };
  }
  return databaseSongCounts(song);
}

function databaseSongCounts(song) {
  const submittedReadableSectionCount = Number(song.submitted_readable_section_count || 0);
  const readableSectionCount = Number(song.readable_section_count || 0);
  const backendHumanCount = Number(song.human_annotation_count || 0);
  const backendAutoCount = Number(song.auto_section_count || 0);
  const deletedSectionCount = Number(song.deleted_section_count || 0);
  const activeAutoSectionCount = Math.max(0, backendAutoCount - deletedSectionCount);
  const backendTotal =
    song.status === "reviewed" && readableSectionCount > 0
      ? readableSectionCount
      : Math.max(readableSectionCount, activeAutoSectionCount, backendHumanCount);
  const actions = Number(song.review_action_count || 0);
  if (submittedReadableSectionCount > 0) {
    return {
      total: backendTotal,
      actions,
      reviewedSections: submittedReadableSectionCount,
      isReviewed: isSongReviewedFromDatabase(song),
    };
  }
  const total = backendTotal;
  const reviewedSections = backendHumanCount;
  return { total, actions, reviewedSections, isReviewed: isSongReviewedFromDatabase(song) };
}

function isSongReviewedFromDatabase(song) {
  return song?.status === "reviewed";
}

function isSongReviewed(song) {
  if (isSongReviewedFromDatabase(song)) return true;
  if (state.selectedSong?.id === song?.id && state.pendingSectionOrderChange) return false;
  return false;
}

async function selectSong(songId, options = {}) {
  if ((state.isSubmittingSong || state.submittingSectionIds.size) && !options.allowDuringSubmit) {
    return showToast("正在提交中，不能切换歌曲", true);
  }
  const song = state.songs.find((item) => item.id === songId);
  if (!song) return showToast("没有找到这首歌", true);
  const selectionToken = state.selectionToken + 1;
  state.selectionToken = selectionToken;
  state.selectedSong = song;
  state.selectedSection = null;
  state.deletedSectionIds = loadDeletedSectionIds(songId);
  state.deletedLocalAnnotationIds = new Set();
  state.confirmedSectionIds = loadConfirmedSectionIds(songId);
  state.pendingSectionOrderChange = false;
  state.currentLyricIndex = null;
  els.songPicker.value = songPickerLabel(song);
  renderSelectedSongCard();
  els.audioPlayer.src = resolveAudioPath(song.audio_path);
  els.audioPlayer.load();

  const { data: runs, error: runError } = await client
    .from("model_runs")
    .select("*")
    .eq("song_id", songId)
    .order("created_at", { ascending: false })
    .limit(1);
  if (selectionToken !== state.selectionToken) return;
  if (runError) return showToast(runError.message, true);
  state.run = runs?.[0] || null;

  if (!state.run) {
    if (selectionToken !== state.selectionToken) return;
    state.sections = [];
    state.annotations = [];
    state.reviews = [];
    state.lyrics = [];
    renderLyricsViewport();
    renderSections();
    return showToast("这首歌还没有 model_run");
  }

  const [
    { data: sections, error: sectionError },
    { data: annotations, error: annotationError },
    { data: reviews, error: reviewError },
    { data: lyrics, error: lyricError },
  ] = await Promise.all([
    client
      .from("auto_sections")
      .select("*")
      .eq("run_id", state.run.id)
      .order("section_index", { ascending: true }),
    client
      .from("human_annotations")
      .select("*")
      .eq("song_id", songId)
      .order("updated_at", { ascending: false }),
    client
      .from("section_reviews")
      .select("*")
      .eq("song_id", songId)
      .order("created_at", { ascending: false }),
    client
      .from("lyric_lines")
      .select("*")
      .eq("song_id", songId)
      .order("line_index", { ascending: true }),
  ]);
  if (selectionToken !== state.selectionToken) return;
  if (sectionError) return showToast(sectionError.message, true);
  if (annotationError) return showToast(annotationError.message, true);
  if (reviewError) return showToast(reviewError.message, true);
  if (lyricError) return showToast(lyricError.message, true);
  state.sections = sections || [];
  state.annotations = annotations || [];
  state.reviews = reviews || [];
  state.lyrics = lyrics || [];
  state.deletedSectionIds = new Set([...state.deletedSectionIds, ...deletedSectionIdsFromReviews(state.reviews)]);
  restoreLocalSectionsFromAnnotations();
  sortSectionsByReviewedOrder();
  await reconcileSelectedSongReviewStatus();
  renderLyricsViewport();
  renderSections();
  renderSummary();
  els.songPicker.value = songPickerLabel(song);
  renderSongDirectory(filteredSongs(els.statusFilter.value, ""));
  if (state.sections.length) selectSection(state.sections[0].id);
}

function songPickerLabel(song) {
  return `${isSongReviewed(song) ? "已质检" : "未质检"} | ${song.title} | ${song.artist || "unknown"} | ${song.id}`;
}

function restoreLocalSectionsFromAnnotations() {
  const localAnnotations = state.annotations.filter((annotation) => !annotation.source_auto_section_id);
  for (const annotation of localAnnotations) {
    state.sections.push({
      id: `local-${annotation.id}`,
      isLocal: true,
      saved_annotation_id: annotation.id,
      local_label: annotation.comment || "新增 Section",
      section_index: Number(annotation.section_index || state.sections.length + 1),
      start_time_sec: Number(annotation.start_time_sec || 0),
      end_time_sec: Number(annotation.end_time_sec || 0),
      section_type: annotation.section_type || "verse",
      lyric_evidence: "人工新增",
      vocal_evidence: "",
      acoustic_evidence: "",
      need_human_review: true,
    });
  }
  state.sections.sort((a, b) => Number(a.section_index) - Number(b.section_index));
}

function sortSectionsByReviewedOrder() {
  state.sections.sort((a, b) => {
    const aFinal = finalResult(a);
    const bFinal = finalResult(b);
    const aIndex = Number(aFinal.section_index || a.section_index || 0);
    const bIndex = Number(bFinal.section_index || b.section_index || 0);
    if (aIndex !== bIndex) return aIndex - bIndex;
    const aStart = Number(aFinal.start_time_sec ?? a.start_time_sec ?? 0);
    const bStart = Number(bFinal.start_time_sec ?? b.start_time_sec ?? 0);
    if (aStart !== bStart) return aStart - bStart;
    return String(a.id).localeCompare(String(b.id));
  });
  renumberSections();
}

function deletedSectionStorageKey(songId) {
  return `section-qc-deleted:${songId}`;
}

function loadDeletedSectionIds(songId) {
  try {
    return new Set(JSON.parse(localStorage.getItem(deletedSectionStorageKey(songId)) || "[]"));
  } catch {
    return new Set();
  }
}

function deletedSectionIdsFromReviews(reviews) {
  const deleted = new Set();
  const seen = new Set();
  for (const review of reviews || []) {
    const autoSectionId = review.auto_section_id;
    if (!autoSectionId || seen.has(autoSectionId)) continue;
    seen.add(autoSectionId);
    const after = reviewAfterJson(review);
    if (after?.deleted === true) deleted.add(autoSectionId);
  }
  return deleted;
}

function reviewAfterJson(review) {
  if (typeof review?.after_json === "string") {
    try {
      return JSON.parse(review.after_json);
    } catch {
      return null;
    }
  }
  return review?.after_json || null;
}

function persistDeletedSectionIds() {
  if (!state.selectedSong) return;
  localStorage.setItem(deletedSectionStorageKey(state.selectedSong.id), JSON.stringify([...state.deletedSectionIds]));
}

function confirmedSectionStorageKey(songId) {
  return `section-qc-confirmed:${songId}`;
}

function loadConfirmedSectionIds(songId) {
  localStorage.removeItem(confirmedSectionStorageKey(songId));
  return new Set();
}

function persistConfirmedSectionIds() {
  if (!state.selectedSong) return;
  localStorage.removeItem(confirmedSectionStorageKey(state.selectedSong.id));
}

function clearConfirmedSectionIds() {
  if (!state.selectedSong) return;
  state.confirmedSectionIds = new Set();
  localStorage.removeItem(confirmedSectionStorageKey(state.selectedSong.id));
}

function markSectionSubmitted(section) {
  if (!section) return;
  state.confirmedSectionIds.delete(section.id);
  persistConfirmedSectionIds();
}

function unmarkSectionSubmitted(section) {
  if (!section) return;
  state.confirmedSectionIds.delete(section.id);
  persistConfirmedSectionIds();
}

async function reconcileSelectedSongReviewStatus() {
  renderSelectedSongCard();
}

function latestAnnotation(section) {
  if (section.isLocal) return state.annotations.find((item) => item.id === section.saved_annotation_id);
  return state.annotations.find((item) => item.source_auto_section_id === section.id);
}

function reviewCountFor(section) {
  const annotation = latestAnnotation(section);
  return state.reviews.filter((review) => {
    const matchesSection = section.isLocal
      ? annotation && review.human_annotation_id === annotation.id
      : review.auto_section_id === section.id;
    return matchesSection && !isAutomaticBoundaryReview(review);
  }).length;
}

function manualReviewCount() {
  return state.reviews.filter((review) => isHumanReviewAction(review)).length;
}

function isSectionSubmitted(section) {
  const annotation = latestAnnotation(section);
  if (annotation) return true;
  return false;
}

function hasHumanReviewForSection(section) {
  const annotation = latestAnnotation(section);
  if (!annotation) return false;
  return state.reviews.some((review) => {
    const matchesSection = section.isLocal
      ? review.human_annotation_id === annotation.id
      : review.auto_section_id === section.id;
    return matchesSection && isHumanReviewAction(review);
  });
}

function isAutomaticBoundaryReview(review) {
  const comment = reviewPayloadComment(review);
  return comment === AUTO_BOUNDARY_SYNC_COMMENT || comment === DELETE_BOUNDARY_SYNC_COMMENT;
}

function isHumanReviewAction(review) {
  if (isAutomaticBoundaryReview(review)) return false;
  return true;
}

function reviewPayloadComment(review) {
  return reviewPayload(review)?.comment || "";
}

function reviewPayload(review) {
  if (typeof review.after_json === "string") {
    try {
      return JSON.parse(review.after_json);
    } catch {
      return null;
    }
  }
  return review.after_json || null;
}

function finalResult(section) {
  const annotation = latestAnnotation(section);
  if (annotation) return { ...annotation, source: "human" };
  return {
    source: section.isLocal ? "local" : "model",
    section_type: section.section_type,
    start_time_sec: section.start_time_sec,
    end_time_sec: section.end_time_sec,
    start_time_label: section.start_time_label,
    end_time_label: section.end_time_label,
    confidence: 1,
    comment: section.isLocal ? section.local_label || "新增 Section" : "默认采用机器标注",
    is_approved: true,
    qc_status: section.isLocal ? "human_modified" : "machine_accepted",
  };
}

function sectionMatchesMachine(section, result) {
  if (!section || section.isLocal || !result) return false;
  return (
    String(result.section_type || "") === String(section.section_type || "") &&
    Math.abs(Number(result.start_time_sec) - Number(section.start_time_sec)) < 0.005 &&
    Math.abs(Number(result.end_time_sec) - Number(section.end_time_sec)) < 0.005
  );
}

function visibleSections() {
  const filter = els.reviewFilter.value;
  return activeSections().filter((section) => {
    const annotation = latestAnnotation(section);
    if (section.isLocal) return true;
    if (filter === "review") {
      return section.need_human_review || Number(section.type_confidence) < 0.74 || Number(section.boundary_confidence) < 0.7;
    }
    if (filter === "approved") return Boolean(annotation);
    return true;
  });
}

function activeSections() {
  return state.sections.filter((section) => !state.deletedSectionIds.has(section.id));
}

function activeSectionIndex(sectionId) {
  return activeSections().findIndex((section) => section.id === sectionId);
}

function sectionOrderSnapshot() {
  return activeSections().map((section) => {
    const final = finalResult(section);
    return {
      id: section.id,
      section_index: Number(section.section_index || 0),
      start_time_sec: Number(final.start_time_sec ?? section.start_time_sec ?? 0),
      end_time_sec: Number(final.end_time_sec ?? section.end_time_sec ?? 0),
    };
  });
}

function reorderSectionsByIds(order) {
  const ids = order.map((item) => (typeof item === "object" ? item.id : item));
  const orderIndex = new Map(ids.map((id, index) => [id, index]));
  state.sections.sort((a, b) => {
    const aIndex = orderIndex.has(a.id) ? orderIndex.get(a.id) : Number.MAX_SAFE_INTEGER;
    const bIndex = orderIndex.has(b.id) ? orderIndex.get(b.id) : Number.MAX_SAFE_INTEGER;
    if (aIndex !== bIndex) return aIndex - bIndex;
    return Number(a.section_index || 0) - Number(b.section_index || 0);
  });
}

function currentTimeSlots() {
  return activeSections().map((section) => {
    const final = finalResult(section);
    return {
      start: Number(final.start_time_sec ?? section.start_time_sec ?? 0),
      end: Number(final.end_time_sec ?? section.end_time_sec ?? 0),
    };
  });
}

function applyTimeSlotsToActiveSections(slots) {
  activeSections().forEach((section, index) => {
    const slot = slots[index];
    if (!slot) return;
    setSectionLocalTiming(section, slot.start, slot.end);
  });
}

function setSectionLocalTiming(section, start, end) {
  section.start_time_sec = start;
  section.end_time_sec = end;
  const annotation = latestAnnotation(section);
  if (annotation) {
    annotation.start_time_sec = start;
    annotation.end_time_sec = end;
  }
}

function applySectionSnapshot(snapshot) {
  reorderSectionsByIds(snapshot);
  const byId = new Map(snapshot.map((item) => [item.id, item]));
  activeSections().forEach((section) => {
    const item = byId.get(section.id);
    if (!item) return;
    section.section_index = Number(item.section_index || section.section_index || 0);
    setSectionLocalTiming(section, Number(item.start_time_sec), Number(item.end_time_sec));
  });
  renumberSections();
}

function insertSectionAfter(section, afterSectionId) {
  const currentIndex = state.sections.findIndex((item) => item.id === section.id);
  if (currentIndex >= 0) state.sections.splice(currentIndex, 1);
  const targetIndex = afterSectionId ? state.sections.findIndex((item) => item.id === afterSectionId) : -1;
  state.sections.splice(targetIndex >= 0 ? targetIndex + 1 : state.sections.length, 0, section);
}

function moveSectionRelativeTo(sectionId, targetSectionId, position = "before") {
  if (sectionId === targetSectionId) return false;
  const previousOrder = sectionOrderSnapshot();
  const slots = currentTimeSlots();
  const fromIndex = state.sections.findIndex((section) => section.id === sectionId);
  const toIndex = state.sections.findIndex((section) => section.id === targetSectionId);
  if (fromIndex < 0 || toIndex < 0) return false;
  const [section] = state.sections.splice(fromIndex, 1);
  const adjustedToIndex = state.sections.findIndex((item) => item.id === targetSectionId);
  state.sections.splice(Math.max(0, adjustedToIndex + (position === "after" ? 1 : 0)), 0, section);
  renumberSections();
  applyTimeSlotsToActiveSections(slots);
  state.pendingSectionOrderChange = true;
  pushUndoEntry({ label: "拖动 section", type: "section-order", order: previousOrder });
  return true;
}

function moveSectionStep(sectionId, direction) {
  const sections = activeSections();
  const index = sections.findIndex((section) => section.id === sectionId);
  const target = sections[index + direction];
  if (!target) return false;
  return moveSectionRelativeTo(sectionId, target.id, direction < 0 ? "before" : "after");
}

function renderSections() {
  const sections = visibleSections();
  els.runMeta.textContent = state.run ? `${state.run.pipeline_version} · ${state.run.model_version}` : "";
  renderTimeline();
  renderSelectedSongCard();
  renderSummary();
  els.sectionList.innerHTML = "";
  els.sectionList.classList.toggle("empty", sections.length === 0);
  if (!sections.length) {
    els.sectionList.textContent = state.run ? "当前过滤条件下没有 section" : "没有模型结果";
    clearInspector();
    return;
  }
  for (const section of sections) {
    const final = finalResult(section);
    const annotation = latestAnnotation(section);
    const submitted = isSectionSubmitted(section);
    const isSubmitting = state.submittingSectionIds.has(section.id);
    const title = sectionCardTitle(section, final, annotation);
    const matchesMachine = sectionMatchesMachine(section, final);
    const chosenSource = matchesMachine || final.qc_status === "machine_accepted" ? "model" : "human";
    const finalStatusText = matchesMachine ? "认同机器" : qcStatusText(final.qc_status || (annotation ? "human_modified" : "machine_accepted"));
    const row = document.createElement("div");
    const visualEffect = state.sectionVisualEffects.get(section.id) || "";
    row.className = `section-row ${section.isLocal ? "local-section" : ""} ${state.selectedSection?.id === section.id ? "active" : ""} ${visualEffect}`;
    row.dataset.sectionId = section.id;
    row.draggable = true;
    row.tabIndex = 0;
    row.innerHTML = `
      <div class="section-move-tools">
        <button type="button" class="section-step section-step-up" aria-label="上移 section" title="上移">↑</button>
        <button type="button" class="section-drag-handle" aria-label="拖动 section 排序" title="拖动排序">☰</button>
        <button type="button" class="section-step section-step-down" aria-label="下移 section" title="下移">↓</button>
      </div>
      <div class="section-index">
        <b>${escapeHtml(title)}</b>
        <span>Section ${section.section_index}${section.isLocal ? " · 新增" : ""} · ${reviewCountFor(section)} 次</span>
      </div>
      <div class="section-stack">
        <div class="result-line model-line ${chosenSource === "model" ? "chosen-line" : ""}">
          <span>${section.isLocal ? "新增" : "机器"}</span>
          <b style="color:${typeColor(section.section_type)}">${escapeHtml(section.section_type)}</b>
          <time>${displayRange(section)}</time>
        </div>
        <div class="result-line ${annotation ? "human-line" : "default-line"} ${chosenSource === "human" ? "chosen-line" : ""}">
          <span>${finalStatusText}</span>
          <b style="color:${typeColor(final.section_type)}">${escapeHtml(final.section_type)}</b>
          <time>${displayRange(final)}</time>
        </div>
      </div>
      <button type="button" class="section-submit ${submitted ? "submitted" : ""} ${isSubmitting ? "submitting" : ""}" ${submitted || isSubmitting ? "disabled" : ""}>${isSubmitting ? "正在提交" : submitted ? "已提交" : "提交"}</button>
    `;
    row.querySelector(".section-submit").addEventListener("click", (event) => {
      event.stopPropagation();
      selectSectionById(section.id, { scroll: false });
      saveAnnotation(annotation ? "edit" : "approve", {
        section,
        values: finalValuesForSection(section),
      });
    });
    row.querySelector(".section-step-up").addEventListener("click", (event) => {
      event.stopPropagation();
      if (moveSectionStep(section.id, -1)) {
        preserveSectionListScroll(() => selectSectionById(section.id, { scroll: false }));
        showToast("已上移 section，未写入数据库，请提交整首歌");
      }
    });
    row.querySelector(".section-step-down").addEventListener("click", (event) => {
      event.stopPropagation();
      if (moveSectionStep(section.id, 1)) {
        preserveSectionListScroll(() => selectSectionById(section.id, { scroll: false }));
        showToast("已下移 section，未写入数据库，请提交整首歌");
      }
    });
    row.addEventListener("click", (event) => {
      if (event.target.closest(".section-submit") || event.target.closest(".section-move-tools")) return;
      previewSection(section.id);
    });
    row.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      previewSection(section.id);
    });
    row.addEventListener("dragstart", (event) => startSectionDrag(event, section.id));
    row.addEventListener("dragover", (event) => moveSectionDragOver(event, section.id));
    row.addEventListener("drop", (event) => dropSectionOn(event, section.id));
    row.addEventListener("dragend", endSectionDrag);
    row.querySelector(".section-drag-handle").addEventListener("pointerdown", (event) => startSectionPointerDrag(event, section.id));
    els.sectionList.appendChild(row);
  }
  if (state.selectedSection) fillStaticInspector(state.selectedSection);
}

function renderTimeline() {
  els.timeline.innerHTML = "";
  if (!state.sections.length) return;
  const sections = state.sections.filter((section) => !state.deletedSectionIds.has(section.id));
  if (!sections.length) return;
  const total = sectionTotalDuration();
  const timelineWidth = els.timeline.clientWidth || els.timeline.getBoundingClientRect().width || 0;
  const track = document.createElement("div");
  track.className = "timeline-track";
  track.style.width = `${state.zoom * 100}%`;
  for (const section of sections) {
    const final = finalResult(section);
    const seg = document.createElement("div");
    seg.className = "timeline-segment";
    seg.dataset.type = final.section_type;
    const start = timeToSeconds(final.start_time_sec, 0);
    const end = timeToSeconds(final.end_time_sec, start);
    const widthPercent = ((end - start) / total) * 100;
    const visualWidthPx = (widthPercent / 100) * timelineWidth * state.zoom;
    if (visualWidthPx < 44) seg.classList.add("compact");
    seg.style.left = `${(start / total) * 100}%`;
    seg.style.width = `${widthPercent}%`;
    const label = document.createElement("span");
    label.className = "timeline-segment-label";
    label.textContent = final.section_type.replace(/_/g, " ");
    seg.appendChild(label);
    seg.title = `${displayRange(final)} ${final.section_type}`;
    seg.addEventListener("click", (event) => {
      event.stopPropagation();
      seekFromTimeline(event, total, true);
    });
    track.appendChild(seg);
  }
  const playhead = document.createElement("div");
  playhead.id = "timelinePlayhead";
  playhead.className = "timeline-playhead";
  track.appendChild(playhead);
  els.timeline.appendChild(track);
  updateTimelinePlayhead();
}

function sectionCardTitle(section, final, annotation) {
  if (section.isLocal && annotation) return `${sectionTypeTitle(final.section_type)}（新增）`;
  if (section.isLocal) return section.local_label || "新增 Section";
  return sectionTypeTitle(final.section_type);
}

function selectSection(sectionId) {
  selectSectionById(sectionId, { scroll: false });
}

function selectSectionById(sectionId, options = {}) {
  const section = state.sections.find((item) => item.id === sectionId);
  if (!section || state.deletedSectionIds.has(section.id)) return;
  state.selectedSection = section;
  fillInspector(section);
  renderSections();
  if (options.scroll !== false) scrollActiveSectionIntoView();
}

function preserveSectionListScroll(callback) {
  const scrollTop = els.sectionList.scrollTop;
  callback();
  els.sectionList.scrollTop = scrollTop;
  requestAnimationFrame(() => {
    els.sectionList.scrollTop = scrollTop;
  });
}

function scrollActiveSectionIntoView() {
  requestAnimationFrame(() => {
    const active = els.sectionList.querySelector(".section-row.active");
    if (!active) return;
    const target =
      active.offsetTop -
      els.sectionList.offsetTop -
      els.sectionList.clientHeight / 2 +
      active.clientHeight / 2;
    els.sectionList.scrollTo({ top: Math.max(0, target), behavior: "smooth" });
  });
}

function flashSectionEffect(sectionId, className, duration = 320) {
  state.sectionVisualEffects.set(sectionId, className);
  renderSections();
  window.setTimeout(() => {
    if (state.sectionVisualEffects.get(sectionId) !== className) return;
    state.sectionVisualEffects.delete(sectionId);
    renderSections();
  }, duration);
}

function removeSectionAfterExit(sectionId, removeCallback) {
  state.sectionVisualEffects.set(sectionId, "section-leaving");
  renderSections();
  window.setTimeout(() => {
    state.sectionVisualEffects.delete(sectionId);
    removeCallback();
  }, 280);
}

function previewSection(sectionId) {
  const section = state.sections.find((item) => item.id === sectionId);
  if (!section || state.deletedSectionIds.has(section.id)) return;
  const final = finalResult(section);
  const playback = final || section;
  const start = timeToSeconds(playback.start_time_sec, 0);
  const end = timeToSeconds(playback.end_time_sec, start + 8);
  state.previewEndTime = Math.min(end, start + 10);
  locatePlaybackAt(start, { sectionId: section.id, play: true, scrollSection: true, keepPreview: true });
}

window.previewSectionById = previewSection;

function locatePlaybackAt(time, options = {}) {
  const target = Math.max(0, timeToSeconds(time, 0));
  if (!options.keepPreview) state.previewEndTime = null;
  const sectionId = options.sectionId || sectionForTime(target)?.id;
  if (sectionId) selectSectionById(sectionId, { scroll: options.scrollSection !== false });
  seekAudioTo(target);
  updateTimelinePlayhead();
  if (options.play) {
    els.audioPlayer.play().catch(() => showToast("音频暂时无法播放", true));
  }
}

function seekAudioTo(time) {
  const target = Math.max(0, timeToSeconds(time, 0));
  if (Number.isFinite(els.audioPlayer.duration) || els.audioPlayer.readyState > 0) {
    els.audioPlayer.currentTime = target;
    return;
  }
  els.audioPlayer.addEventListener(
    "loadedmetadata",
    () => {
      els.audioPlayer.currentTime = target;
      updateTimelinePlayhead();
    },
    { once: true },
  );
}

function fillInspector(section) {
  const final = finalResult(section);
  fillStaticInspector(section);
  els.editType.value = final.section_type;
  els.editStart.value = displayTime(final.start_time_sec);
  els.editEnd.value = displayTime(final.end_time_sec);
  markEditorSnapshotClean();
  renderTypeButtonState();
}

function fillStaticInspector(section) {
  const final = finalResult(section);
  els.selectedBadge.textContent = `Section ${section.section_index}`;
  els.finalResultSummary.innerHTML = `
    <strong>${sectionTypeTitle(final.section_type)} · ${displayRange(final)}</strong>
    <span>${escapeHtml(final.section_type)} · ${displayRange(final)} · 已质检 ${reviewCountFor(section)} 次</span>
  `;
  els.boundaryConfidence.textContent = numberText(section.boundary_confidence);
  els.typeConfidence.textContent = numberText(section.type_confidence);
  els.reviewFlag.textContent = section.need_human_review ? "true" : "false";
  els.lyricEvidence.textContent = section.lyric_evidence || "-";
  els.vocalEvidence.textContent = section.vocal_evidence || "-";
  els.acousticEvidence.textContent = section.acoustic_evidence || "-";
  updateActiveLyric();
  renderTypeButtonState();
}

function clearInspector() {
  state.selectedSection = null;
  state.lastEditorSnapshotKey = "";
  els.selectedBadge.textContent = "未选择";
  els.finalResultSummary.textContent = "选择左侧 section 后编辑";
  els.editForm.reset();
  els.boundaryConfidence.textContent = "-";
  els.typeConfidence.textContent = "-";
  els.reviewFlag.textContent = "-";
  els.lyricEvidence.textContent = "-";
  els.vocalEvidence.textContent = "-";
  els.acousticEvidence.textContent = "-";
  renderTypeButtonState();
}

function renderTypeButtonState() {
  for (const button of els.typeQuickButtons.querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.type === els.editType.value);
  }
}

function editorSnapshot(section = state.selectedSection) {
  if (!section) return null;
  return {
    sectionId: section.id,
    type: els.editType.value,
    start: els.editStart.value,
    end: els.editEnd.value,
  };
}

function editorSnapshotKey(snapshot) {
  if (!snapshot) return "";
  return `${snapshot.sectionId}|${snapshot.type}|${snapshot.start}|${snapshot.end}`;
}

function rememberEditorState(label = "编辑") {
  const snapshot = editorSnapshot();
  const key = editorSnapshotKey(snapshot);
  if (!snapshot || key === state.lastEditorSnapshotKey) return;
  pushUndoEntry({ label, type: "editor", snapshot });
  state.lastEditorSnapshotKey = key;
}

function markEditorSnapshotClean() {
  state.lastEditorSnapshotKey = editorSnapshotKey(editorSnapshot());
}

function pushUndoEntry(entry) {
  state.undoStack.push(entry);
  if (state.undoStack.length > 10) state.undoStack.shift();
}

function applyEditorSnapshot(snapshot) {
  const section = state.sections.find((item) => item.id === snapshot.sectionId);
  if (!section || state.deletedSectionIds.has(section.id)) return false;
  state.selectedSection = section;
  els.editType.value = snapshot.type;
  els.editStart.value = snapshot.start;
  els.editEnd.value = snapshot.end;
  fillStaticInspector(section);
  renderTypeButtonState();
  renderSections();
  markEditorSnapshotClean();
  return true;
}

function undoLastEditorAction() {
  while (state.undoStack.length) {
    const entry = state.undoStack.pop();
    if (applyUndoEntry(entry)) {
      showToast(`已撤销：${entry.label}`);
      return;
    }
  }
  showToast("没有可撤销的操作");
}

function applyUndoEntry(entry) {
  if (entry.type === "editor") return applyEditorSnapshot(entry.snapshot);
  if (entry.type === "remove-local-section") return undoLocalSectionAdd(entry);
  if (entry.type === "section-order") return applySectionOrderSnapshot(entry);
  return false;
}

function undoLocalSectionAdd(entry) {
  const section = state.sections.find((item) => item.id === entry.sectionId);
  if (!section || !section.isLocal || section.saved_annotation_id) return false;
  state.sections = state.sections.filter((item) => item.id !== entry.sectionId);
  renumberSections();
  const fallback =
    visibleSections().find((item) => item.id === entry.previousSelectedSectionId) ||
    visibleSections()[Math.max(0, Number(entry.insertIndex || 1) - 1)];
  if (fallback) selectSectionById(fallback.id, { scroll: true });
  else {
    clearInspector();
    renderSections();
  }
  return true;
}

function applySectionOrderSnapshot(entry) {
  if (!Array.isArray(entry.order) || !entry.order.length) return false;
  applySectionSnapshot(entry.order);
  state.pendingSectionOrderChange = true;
  renderSections();
  if (state.selectedSection && !state.deletedSectionIds.has(state.selectedSection.id)) {
    selectSectionById(state.selectedSection.id, { scroll: false });
  }
  return true;
}

function currentEditorValues() {
  return {
    type: els.editType.value,
    start: parseTimeInput(els.editStart.value),
    end: parseTimeInput(els.editEnd.value),
  };
}

function finalValuesForSection(section) {
  const final = finalResult(section);
  return {
    type: final.section_type,
    start: Number(final.start_time_sec),
    end: Number(final.end_time_sec),
  };
}

function songEndTime() {
  const databaseDuration = Number(state.selectedSong?.duration_sec || 0);
  if (databaseDuration > 0) return databaseDuration;
  const audioDuration = Number(els.audioPlayer.duration || 0);
  if (audioDuration > 0) return audioDuration;
  return Math.max(...activeSections().map((section) => Number(finalResult(section).end_time_sec || 0)), 0);
}

function wholeSongContinuityIssue() {
  const sections = activeSections();
  if (!sections.length) return "当前歌曲没有可提交的 section";
  const first = finalResult(sections[0]);
  if (Math.abs(Number(first.start_time_sec || 0)) > TIME_EPSILON) {
    return `第一段必须从 0:00 开始，现在是 ${displayTime(first.start_time_sec)}`;
  }
  for (let index = 1; index < sections.length; index += 1) {
    const previous = finalResult(sections[index - 1]);
    const current = finalResult(sections[index]);
    const previousEnd = Number(previous.end_time_sec || 0);
    const currentStart = Number(current.start_time_sec || 0);
    if (Math.abs(previousEnd - currentStart) > TIME_EPSILON) {
      return `Section ${index} 和 Section ${index + 1} 时间不连续：${displayTime(previousEnd)} / ${displayTime(currentStart)}`;
    }
  }
  const last = finalResult(sections.at(-1));
  const songEnd = songEndTime();
  if (Math.abs(Number(last.end_time_sec || 0) - songEnd) > TIME_EPSILON) {
    return `最后一段必须连接到歌曲末尾 ${displayTime(songEnd)}，现在结束于 ${displayTime(last.end_time_sec)}`;
  }
  return "";
}

function submitContextSongId(context) {
  return context?.songId || state.selectedSong?.id;
}

function submitContextRunId(context) {
  return context?.runId || state.run?.id;
}

function createSubmitContext(sections) {
  if (!state.selectedSong || !state.run) throw new Error("先选择一首有模型结果的歌");
  const songId = state.selectedSong.id;
  const runId = state.run.id;
  if (state.run.song_id && state.run.song_id !== songId) {
    throw new Error("当前歌曲和 model_run 不一致，已阻止写入数据库");
  }
  for (const section of sections) {
    if (!section.isLocal && section.run_id && section.run_id !== runId) {
      throw new Error(`Section ${section.section_index || ""} 不属于当前 model_run，已阻止写入数据库`);
    }
  }
  return {
    songId,
    runId,
    selectionToken: state.selectionToken,
    expectedSectionCount: sections.length,
  };
}

function assertSubmitContext(context) {
  if (!context) return;
  if (state.selectionToken !== context.selectionToken) {
    throw new Error("提交期间歌曲选择发生变化，已停止写入数据库");
  }
  if (state.selectedSong?.id !== context.songId || state.run?.id !== context.runId) {
    throw new Error("提交期间歌曲或 model_run 发生变化，已停止写入数据库");
  }
}

function assertWrittenAnnotation(annotation, section, context) {
  if (!annotation) throw new Error("数据库没有返回 human_annotation");
  if (annotation.song_id !== context.songId) {
    throw new Error("数据库写入校验失败：human_annotations.song_id 与当前歌曲不一致");
  }
  if (annotation.source_run_id && annotation.source_run_id !== context.runId) {
    throw new Error("数据库写入校验失败：human_annotations.source_run_id 与当前 run 不一致");
  }
  const expectedAutoSectionId = section.isLocal ? null : section.id;
  if ((annotation.source_auto_section_id || null) !== expectedAutoSectionId) {
    throw new Error("数据库写入校验失败：human_annotations.source_auto_section_id 与当前 section 不一致");
  }
}

function buildPayload(section, values, context = null) {
  return {
    song_id: submitContextSongId(context),
    source_run_id: submitContextRunId(context),
    source_auto_section_id: section.isLocal ? null : section.id,
    annotator: "local_reviewer",
    section_index: section.section_index,
    start_time_sec: values.start,
    end_time_sec: values.end,
    section_type: values.type,
    confidence: values.confidence,
    is_approved: values.action !== "unsure",
    qc_status: values.qcStatus,
    comment: values.comment,
  };
}

function valuesMatchMachine(section, values) {
  if (!section || section.isLocal || !values) return false;
  return (
    String(values.type || "") === String(section.section_type || "") &&
    Math.abs(Number(values.start) - Number(section.start_time_sec)) < 0.005 &&
    Math.abs(Number(values.end) - Number(section.end_time_sec)) < 0.005
  );
}

function valuesMatchCurrentAnnotation(section, values) {
  const annotation = latestAnnotation(section);
  if (!annotation || !values) return false;
  return (
    String(values.type || "") === String(annotation.section_type || "") &&
    Math.abs(Number(values.start) - Number(annotation.start_time_sec)) < 0.005 &&
    Math.abs(Number(values.end) - Number(annotation.end_time_sec)) < 0.005
  );
}

function shouldPersistHumanAnnotation(section, values) {
  return true;
}

async function upsertAnnotation(section, payload, action, context = null) {
  if (context) assertSubmitContext(context);
  const before = latestAnnotation(section) || section;
  const data = await persistHumanAnnotation(section, payload);
  if (context) assertWrittenAnnotation(data, section, context);

  const reviewPayload = {
    song_id: submitContextSongId(context),
    run_id: submitContextRunId(context),
    auto_section_id: section.isLocal ? null : section.id,
    human_annotation_id: data.id,
    action,
    reviewer: "local_reviewer",
    before_json: before,
    after_json: payload,
  };
  const { data: reviewData, error: reviewError } = await withSubmitTimeout(
    client.from("section_reviews").insert(reviewPayload).select().single(),
    "section 记录提交",
  );
  if (reviewError) throw reviewError;
  if (context) {
    if (reviewData.song_id !== context.songId || reviewData.run_id !== context.runId) {
      throw new Error("数据库写入校验失败：section_reviews 与当前歌曲/run 不一致");
    }
    const expectedAutoSectionId = section.isLocal ? null : section.id;
    if ((reviewData.auto_section_id || null) !== expectedAutoSectionId) {
      throw new Error("数据库写入校验失败：section_reviews.auto_section_id 与当前 section 不一致");
    }
  }

  Object.defineProperty(data, "_review_id", {
    value: reviewData.id,
    enumerable: false,
    configurable: true,
  });
  state.annotations = [
    data,
    ...state.annotations.filter((item) => {
      if (section.isLocal) return item.id !== data.id;
      return item.source_auto_section_id !== section.id;
    }),
  ];
  state.reviews = [reviewData, ...state.reviews];
  return data;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function verifyReadableAnnotation(annotationId) {
  if (!annotationId) throw new Error("缺少可视化校验所需的 human_annotation_id");
  for (let attempt = 0; attempt < 4; attempt += 1) {
    const { data, error } = await withSubmitTimeout(
      client
        .from("human_annotations_readable")
        .select("human_annotation_id")
        .eq("human_annotation_id", annotationId)
        .limit(1),
      "可视化视图校验",
    );
    if (error) throw error;
    if ((data || []).length > 0) return true;
    await wait(350);
  }
  throw new Error("数据库已写入，但 human_annotations_readable 还没有读到对应 section，请刷新 Supabase 视图后再继续");
}

async function verifyReadableAnnotations(writtenAnnotations) {
  const annotationIds = [
    ...new Set(
      writtenAnnotations
        .map((item) => item?.annotation?.id)
        .filter(Boolean),
    ),
  ];
  await Promise.all(annotationIds.map((annotationId) => verifyReadableAnnotation(annotationId)));
}

async function verifyAnnotationWrite(annotationId, expected = null, reviewId = null) {
  if (!annotationId) throw new Error("缺少提交校验所需的 human_annotation_id");
  for (let attempt = 0; attempt < 4; attempt += 1) {
    const reviewQuery = reviewId
      ? client
          .from("section_reviews")
          .select("id,song_id,run_id,auto_section_id,human_annotation_id")
          .eq("id", reviewId)
          .limit(1)
      : client
          .from("section_reviews")
          .select("id,song_id,run_id,auto_section_id,human_annotation_id")
          .eq("human_annotation_id", annotationId)
          .order("created_at", { ascending: false })
          .limit(10);
    const [{ data: annotations, error: annotationError }, { data: reviews, error: reviewError }] = await Promise.all([
      withSubmitTimeout(
        client
          .from("human_annotations")
          .select("id,song_id,source_run_id,source_auto_section_id")
          .eq("id", annotationId)
          .limit(1),
        "人工标注写入校验",
      ),
      withSubmitTimeout(reviewQuery, "质检记录写入校验"),
    ]);
    if (annotationError) throw annotationError;
    if (reviewError) throw reviewError;
    const annotation = annotations?.[0];
    const matchingReview = (reviews || []).find((review) => {
      if (!expected) return true;
      return (
        review.song_id === expected.songId &&
        review.run_id === expected.runId &&
        (review.auto_section_id || null) === (expected.autoSectionId || null)
      );
    });
    const annotationMatches =
      annotation &&
      (!expected ||
        (annotation.song_id === expected.songId &&
          (!annotation.source_run_id || annotation.source_run_id === expected.runId) &&
          (annotation.source_auto_section_id || null) === (expected.autoSectionId || null)));
    if (annotationMatches && matchingReview) return true;
    await wait(350);
  }
  throw new Error(
    reviewId
      ? `数据库写入校验失败：没有读到刚写入的 section_reviews 记录 ${reviewId}`
      : "数据库写入校验失败：human_annotations 或 section_reviews 没有读到对应记录",
  );
}

async function verifyReadableSong(songId, expectedSectionCount) {
  for (let attempt = 0; attempt < 4; attempt += 1) {
    const [{ data: rows, error: readableError }, { data: songs, error: songError }] = await Promise.all([
      withSubmitTimeout(
        client.from("human_annotations_readable").select("human_annotation_id").eq("song_id", songId),
        "可视化视图校验",
      ),
      withSubmitTimeout(client.from("songs").select("status").eq("id", songId).limit(1), "歌曲状态校验"),
    ]);
    if (readableError) throw readableError;
    if (songError) throw songError;
    const statusOk = songs?.[0]?.status === "reviewed";
    const readableRows = rows || [];
    const actualSections = readableRows.filter((row) => row.human_annotation_id).length;
    if (statusOk && actualSections >= expectedSectionCount) return true;
    await wait(500);
  }
  throw new Error(
    `整首歌已提交但可视化视图未校验通过：human_annotations_readable 行数不足，预期 ${expectedSectionCount} 段`,
  );
}

async function verifyNoReadableSongRows(songId) {
  for (let attempt = 0; attempt < 4; attempt += 1) {
    const { data, error } = await withSubmitTimeout(
      client.from("human_annotations_readable").select("human_annotation_id").eq("song_id", songId),
      "可视化视图清空校验",
    );
    if (error) throw error;
    if (!(data || []).some((row) => row.human_annotation_id)) return true;
    await wait(350);
  }
  throw new Error("数据库清空校验失败：human_annotations_readable 仍能读到这首歌的 section");
}

async function verifyDeletedSectionReview(context, autoSectionId) {
  for (let attempt = 0; attempt < 4; attempt += 1) {
    const { data, error } = await withSubmitTimeout(
      client
        .from("section_reviews")
        .select("id,song_id,run_id,auto_section_id,action,after_json")
        .eq("song_id", context.songId)
        .eq("run_id", context.runId)
        .eq("auto_section_id", autoSectionId)
        .order("created_at", { ascending: false })
        .limit(5),
      "删除记录回读校验",
    );
    if (error) throw error;
    if ((data || []).some((review) => reviewAfterJson(review)?.deleted === true)) return true;
    await wait(350);
  }
  throw new Error("数据库写入校验失败：没有读到 section 删除记录");
}

async function persistHumanAnnotation(section, payload) {
  const dbPayload = state.supportsQcStatus ? payload : withoutQcStatus(payload);
  let response;
  if (section.isLocal && section.saved_annotation_id) {
    response = await withSubmitTimeout(
      client.from("human_annotations").update(dbPayload).eq("id", section.saved_annotation_id).select().single(),
      "人工标注提交",
    );
  } else if (section.isLocal) {
    response = await withSubmitTimeout(
      client.from("human_annotations").insert(dbPayload).select().single(),
      "人工标注提交",
    );
  } else {
    response = await withSubmitTimeout(
      client
        .from("human_annotations")
        .upsert(dbPayload, { onConflict: "source_auto_section_id" })
        .select()
        .single(),
      "人工标注提交",
    );
  }
  if (response.error && state.supportsQcStatus && /qc_status|schema cache/i.test(response.error.message || "")) {
    state.supportsQcStatus = false;
    return persistHumanAnnotation(section, payload);
  }
  if (response.error) throw response.error;
  if (section.isLocal) section.saved_annotation_id = response.data.id;
  return response.data;
}

function withoutQcStatus(payload) {
  const { qc_status: _qcStatus, ...rest } = payload;
  return rest;
}

async function saveAnnotation(action = "edit", options = {}) {
  const section = options.section || state.selectedSection;
  if (!section || !state.selectedSong || !state.run) return showToast("先选择一个 section", true);
  if (state.isSubmittingSong) return showToast("整首歌正在提交中");
  if (state.submittingSectionIds.has(section.id)) return showToast("本段正在提交中");
  setSubmitBusy(true);
  let start;
  let end;
  let type;
  try {
    const values = options.values || currentEditorValues();
    start = values.start;
    end = values.end;
    type = values.type;
  } catch (error) {
    setSubmitBusy(false);
    return showToast(error.message, true);
  }
  if (end <= start) {
    setSubmitBusy(false);
    return showToast("结束时间必须晚于开始时间", true);
  }
  const previousSection = previousSectionBefore(section);
  const previousFinal = previousSection ? finalResult(previousSection) : null;
  if (previousFinal && start <= Number(previousFinal.start_time_sec)) {
    setSubmitBusy(false);
    return showToast("当前 section 的开始时间必须晚于上一段开始时间", true);
  }
  const nextSection = nextSectionAfter(section);
  const nextFinal = nextSection ? finalResult(nextSection) : null;
  if (nextFinal && end >= Number(nextFinal.end_time_sec)) {
    setSubmitBusy(false);
    return showToast("当前 section 的结束时间必须早于下一段结束时间", true);
  }
  const context = createSubmitContext(activeSections());
  const expected = {
    songId: context.songId,
    runId: context.runId,
    autoSectionId: section.isLocal ? null : section.id,
  };
  const payload = buildPayload(section, {
    start,
    end,
    type,
    confidence: 1,
    comment: sectionCommentForAction(section, action, { start, end, type }),
    qcStatus: qcStatusForAction(section, action, { start, end, type }),
    action,
  }, context);
  let submitFailureLabel = "本段提交失败";
  try {
    state.submittingSectionIds.add(section.id);
    renderSections();
    showToast("正在提交本段", false, { persistent: true });
    const writtenAnnotations = [];
    if (shouldPersistHumanAnnotation(section, { start, end, type })) {
      writtenAnnotations.push({ annotation: await upsertAnnotation(section, payload, action, context), expected });
    }
    const previousSync = await syncPreviousSectionEnd(section, start, context);
    const nextSync = await syncNextSectionStart(section, end, context);
    if (previousSync) writtenAnnotations.push(previousSync);
    if (nextSync) writtenAnnotations.push(nextSync);
    submitFailureLabel = "本段数据库校验失败";
    await Promise.all(
      writtenAnnotations.map(({ annotation, expected: itemExpected }) =>
        verifyAnnotationWrite(annotation.id, itemExpected, annotation._review_id),
      ),
    );
    submitFailureLabel = "本段可视化校验失败";
    await verifyReadableAnnotations(writtenAnnotations);
    markSectionSubmitted(section);
    renderSections();
    const isCompleteAfterSectionSubmit = isCurrentSongComplete();
    const continuityIssue = isCompleteAfterSectionSubmit ? wholeSongContinuityIssue() : "";
    submitFailureLabel = "歌曲状态更新失败";
    await updateSongStatus("in_review", context.songId);
    await withSubmitTimeout(loadSongs(), "刷新提交结果");
    if (isCompleteAfterSectionSubmit) {
      if (continuityIssue) {
        showToast(`本段提交成功，整首歌未提交：${continuityIssue}`, true);
        renderSections();
        return;
      }
      showToast("本段提交成功，请手动提交整首歌");
    } else {
      showToast("本段提交成功");
    }
    renderSections();
  } catch (error) {
    showToast(submitErrorMessage(error, submitFailureLabel), true);
  } finally {
    state.submittingSectionIds.delete(section.id);
    setWholeSongSubmitBusy(false);
    setSubmitBusy(false);
    renderSections();
  }
}

function areAllSectionsSubmitted() {
  const sections = activeSections();
  return sections.length > 0 && sections.every((section) => isSectionSubmitted(section));
}

function firstUnsubmittedSection() {
  return activeSections().find((section) => !isSectionSubmitted(section)) || null;
}

function isCurrentSongComplete() {
  return areAllSectionsSubmitted();
}

function adjacentSection(section, offset) {
  const sections = state.sections.filter((item) => !state.deletedSectionIds.has(item.id));
  const index = sections.findIndex((item) => item.id === section.id);
  return index >= 0 ? sections[index + offset] || null : null;
}

function previousSectionBefore(section) {
  return adjacentSection(section, -1);
}

function nextSectionAfter(section) {
  return adjacentSection(section, 1);
}

async function syncPreviousSectionEnd(section, end, context = null) {
  const previous = previousSectionBefore(section);
  if (!previous) return;
  const final = finalResult(previous);
  if (Math.abs(Number(final.end_time_sec) - end) < 0.005) return;
  const payload = buildPayload(previous, {
    start: Number(final.start_time_sec),
    end,
    type: final.section_type,
    confidence: Number(final.confidence ?? 1),
    comment: AUTO_BOUNDARY_SYNC_COMMENT,
    qcStatus: "human_modified",
    action: "edit",
  }, context);
  return {
    annotation: await upsertAnnotation(previous, payload, "edit", context),
    expected: {
      songId: submitContextSongId(context),
      runId: submitContextRunId(context),
      autoSectionId: previous.isLocal ? null : previous.id,
    },
  };
}

async function syncNextSectionStart(section, start, context = null) {
  const next = nextSectionAfter(section);
  if (!next) return;
  const final = finalResult(next);
  if (Math.abs(Number(final.start_time_sec) - start) < 0.005) return;
  const payload = buildPayload(next, {
    start,
    end: Number(final.end_time_sec),
    type: final.section_type,
    confidence: Number(final.confidence ?? 1),
    comment: AUTO_BOUNDARY_SYNC_COMMENT,
    qcStatus: "human_modified",
    action: "edit",
  }, context);
  return {
    annotation: await upsertAnnotation(next, payload, "edit", context),
    expected: {
      songId: submitContextSongId(context),
      runId: submitContextRunId(context),
      autoSectionId: next.isLocal ? null : next.id,
    },
  };
}

async function upsertBoundarySync(section, { start = null, end = null, comment = AUTO_BOUNDARY_SYNC_COMMENT, context = null } = {}) {
  if (!section) return null;
  const final = finalResult(section);
  const nextStart = start === null ? Number(final.start_time_sec) : Number(start);
  const nextEnd = end === null ? Number(final.end_time_sec) : Number(end);
  if (
    Math.abs(Number(final.start_time_sec) - nextStart) < TIME_EPSILON &&
    Math.abs(Number(final.end_time_sec) - nextEnd) < TIME_EPSILON
  ) {
    return null;
  }
  const payload = buildPayload(section, {
    start: nextStart,
    end: nextEnd,
    type: final.section_type,
    confidence: Number(final.confidence ?? 1),
    comment,
    qcStatus: "human_modified",
    action: "edit",
  }, context);
  return upsertAnnotation(section, payload, "edit", context);
}

async function connectAroundDeletedSection(section, context = null) {
  const previous = previousSectionBefore(section);
  const next = nextSectionAfter(section);
  if (previous && next) {
    const nextFinal = finalResult(next);
    const annotation = await upsertBoundarySync(previous, {
      end: Number(nextFinal.start_time_sec),
      comment: DELETE_BOUNDARY_SYNC_COMMENT,
      context,
    });
    return annotation ? { section: previous, annotation } : { section: previous, annotation: null };
  }
  if (previous && !next) {
    const annotation = await upsertBoundarySync(previous, {
      end: songEndTime(),
      comment: DELETE_BOUNDARY_SYNC_COMMENT,
      context,
    });
    return annotation ? { section: previous, annotation } : { section: previous, annotation: null };
  }
  if (!previous && next) {
    const annotation = await upsertBoundarySync(next, {
      start: 0,
      comment: DELETE_BOUNDARY_SYNC_COMMENT,
      context,
    });
    return annotation ? { section: next, annotation } : { section: next, annotation: null };
  }
  return null;
}

function setSubmitBusy(isBusy) {
  els.saveEdit.disabled = isBusy;
  els.saveEdit.textContent = isBusy ? "正在提交..." : "提交本段";
}

function setWholeSongSubmitBusy(isBusy) {
  state.isSubmittingSong = isBusy;
  els.finishReview.disabled = isBusy;
  els.finishReview.textContent = isBusy ? "正在提交..." : "提交整首歌";
  if (els.markPendingReview) {
    els.markPendingReview.disabled = isBusy;
    els.markPendingReview.textContent = isBusy ? "正在提交..." : "标为未质检";
  }
}

function qcStatusForAction(section, action, values = null) {
  if (!section.isLocal && (action === "approve" || valuesMatchMachine(section, values))) return "machine_accepted";
  return "human_modified";
}

function sectionCommentForAction(section, action, values = null) {
  if (section.isLocal) return section.local_label || "新增 Section";
  if (action === "approve" || valuesMatchMachine(section, values)) return "默认采用机器标注";
  if (action === "unsure") return "人工标记待定";
  const comment = latestAnnotation(section)?.comment;
  return comment && comment !== AUTO_BOUNDARY_SYNC_COMMENT ? comment : "人工修改";
}

async function writeSections(sections, finalStatus, context = null) {
  if (!state.selectedSong || !state.run) throw new Error("先选择一首有模型结果的歌");
  if (finalStatus === "reviewed" && !state.manualWholeSongSubmitActive) {
    throw new Error("整首歌 reviewed 状态只能通过提交整首歌按钮写入");
  }
  const submitContext = context || createSubmitContext(sections);
  const sectionPlans = sections.map((section) => {
    const final = finalResult(section);
    const values = {
      type: final.section_type,
      start: Number(final.start_time_sec),
      end: Number(final.end_time_sec),
    };
    const action = section.isLocal || !valuesMatchMachine(section, values) ? "edit" : "approve";
    return {
      section,
      action,
      expected: {
        songId: submitContext.songId,
        runId: submitContext.runId,
        autoSectionId: section.isLocal ? null : section.id,
      },
      payload: buildPayload(
        section,
        {
          ...values,
          confidence: Number(final.confidence ?? 1),
          comment: sectionCommentForAction(section, action, values),
          qcStatus: qcStatusForAction(section, action, values),
          action,
        },
        submitContext,
      ),
    };
  });
  const writtenAnnotations = [];
  for (const plan of sectionPlans) {
    assertSubmitContext(submitContext);
    const annotation = await upsertAnnotation(plan.section, plan.payload, plan.action, submitContext);
    await verifyAnnotationWrite(annotation.id, plan.expected, annotation._review_id);
    writtenAnnotations.push({ annotation, expected: plan.expected });
    markSectionSubmitted(plan.section);
  }
  await updateSongStatus(finalStatus, submitContext.songId);
  await Promise.all(writtenAnnotations.map(({ annotation }) => verifyReadableAnnotation(annotation.id)));
  await verifyReadableSong(submitContext.songId, sections.length);
  state.pendingSectionOrderChange = false;
  state.deletedLocalAnnotationIds = new Set();
  await withSubmitTimeout(loadSongs(), "刷新提交结果");
  await withSubmitTimeout(selectSong(submitContext.songId, { allowDuringSubmit: true }), "刷新提交结果");
  renderSections();
}

async function finalizeWholeSongReview() {
  const sections = activeSections();
  const context = createSubmitContext(sections);
  await deleteDeletedAnnotations(context);
  await writeSections(sections, "reviewed", context);
}

async function finishReview() {
  if (state.isSubmittingSong) return;
  if (!state.selectedSong || !state.run) return showToast("先选择一首有模型结果的歌", true);
  if (!activeSections().length) return showToast("当前歌曲没有可提交的 section", true);
  const unsubmittedSection = firstUnsubmittedSection();
  if (unsubmittedSection) {
    selectSectionById(unsubmittedSection.id, { scroll: true });
    return showToast("还有 section 未提交，已跳到第一个未提交 section", true);
  }
  const continuityIssue = wholeSongContinuityIssue();
  if (continuityIssue) return showToast(continuityIssue, true);
  try {
    setWholeSongSubmitBusy(true);
    state.manualWholeSongSubmitActive = true;
    showToast("正在提交整首歌", false, { persistent: true });
    const sections = activeSections();
    const context = createSubmitContext(sections);
    await deleteDeletedAnnotations(context);
    await writeSections(sections, "reviewed", context);
    showToast("整首歌提交成功");
  } catch (error) {
    showToast(submitErrorMessage(error, "整首歌提交失败"), true);
  } finally {
    state.manualWholeSongSubmitActive = false;
    setWholeSongSubmitBusy(false);
  }
}

async function resetCurrentSection() {
  const section = state.selectedSection;
  if (!section || !state.selectedSong) return showToast("先选择一个 section", true);
  const songId = state.selectedSong.id;
  const targetSectionId = section.isLocal ? null : section.id;
  try {
    setResetBusy(true);
    showToast("当前 section 重置正在提交", false, { persistent: true });
    if (section.isLocal) {
      if (section.saved_annotation_id) {
        await deleteReviewsByHumanAnnotationId(section.saved_annotation_id);
        await deleteHumanAnnotationById(section.saved_annotation_id);
      }
      state.sections = state.sections.filter((item) => item.id !== section.id);
      renumberSections();
    } else {
      const annotationIds = humanAnnotationIdsForSection(section);
      await deleteReviewsByAutoSection(section.id);
      await deleteReviewsByHumanAnnotationIds(annotationIds);
      await deleteHumanAnnotationsByAutoSection(section.id);
      state.deletedSectionIds.delete(section.id);
      persistDeletedSectionIds();
    }
    unmarkSectionSubmitted(section);
    await updateSongStatus("in_review", songId);
    await withSubmitTimeout(loadSongs(), "刷新提交结果");
    await withSubmitTimeout(selectSong(songId), "刷新提交结果");
    if (targetSectionId && state.sections.some((item) => item.id === targetSectionId)) {
      selectSectionById(targetSectionId, { scroll: false });
    } else if (state.sections[0]) {
      selectSectionById(state.sections[0].id, { scroll: false });
    } else {
      clearInspector();
    }
    showToast("当前 section 重置提交成功");
  } catch (error) {
    showToast(submitErrorMessage(error, "当前 section 重置提交失败"), true);
  } finally {
    setResetBusy(false);
  }
}

async function resetWholeSong() {
  if (!state.selectedSong) return showToast("先选择一首歌", true);
  const songId = state.selectedSong.id;
  try {
    setResetBusy(true);
    showToast("整首歌重置正在提交", false, { persistent: true });
    await deleteReviewsBySong(songId);
    await deleteHumanAnnotationsBySong(songId);
    await verifyNoReadableSongRows(songId);
    state.annotations = [];
    state.reviews = [];
    state.deletedSectionIds = new Set();
    persistDeletedSectionIds();
    clearConfirmedSectionIds();
    state.pendingSectionOrderChange = false;
    state.sections = state.sections.filter((section) => !section.isLocal);
    renumberSections();
    await updateSongStatus("pending_review", songId);
    await withSubmitTimeout(loadSongs(), "刷新提交结果");
    await withSubmitTimeout(selectSong(songId), "刷新提交结果");
    if (state.sections[0]) selectSectionById(state.sections[0].id, { scroll: false });
    else clearInspector();
    renderSections();
    showToast("整首歌重置提交成功");
  } catch (error) {
    showToast(submitErrorMessage(error, "整首歌重置提交失败"), true);
  } finally {
    setResetBusy(false);
  }
}

async function markSongPendingReviewOnly() {
  if (state.isSubmittingSong) return;
  if (!state.selectedSong) return showToast("先选择一首歌", true);
  const songId = state.selectedSong.id;
  try {
    setWholeSongSubmitBusy(true);
    showToast("状态改为未质检正在提交", false, { persistent: true });
    await deleteReviewsBySong(songId);
    await deleteHumanAnnotationsBySong(songId);
    await verifyNoReadableSongRows(songId);
    state.annotations = [];
    state.reviews = [];
    state.deletedSectionIds = new Set();
    persistDeletedSectionIds();
    clearConfirmedSectionIds();
    state.pendingSectionOrderChange = false;
    state.sections = state.sections.filter((section) => !section.isLocal);
    renumberSections();
    await updateSongStatus("pending_review", songId);
    await withSubmitTimeout(loadSongs(), "刷新提交结果");
    await withSubmitTimeout(selectSong(songId, { allowDuringSubmit: true }), "刷新提交结果");
    renderSections();
    showToast("状态改为未质检提交成功");
  } catch (error) {
    showToast(submitErrorMessage(error, "状态改为未质检提交失败"), true);
  } finally {
    setWholeSongSubmitBusy(false);
  }
}

function humanAnnotationIdsForSection(section) {
  return state.annotations
    .filter((item) => item.source_auto_section_id === section.id || (section.isLocal && item.id === section.saved_annotation_id))
    .map((item) => item.id)
    .filter(Boolean);
}

function setResetBusy(isBusy) {
  for (const button of [els.resetCurrentSection, els.resetSong].filter(Boolean)) {
    button.disabled = isBusy;
  }
  if (els.resetCurrentSection) els.resetCurrentSection.textContent = isBusy ? "提交中..." : "重置当前section";
  if (els.resetSong) els.resetSong.textContent = isBusy ? "提交中..." : "重置整首歌";
}

async function deleteHumanAnnotationsByAutoSection(autoSectionId, songId = state.selectedSong?.id) {
  let query = client.from("human_annotations").delete().eq("source_auto_section_id", autoSectionId);
  if (songId) query = query.eq("song_id", songId);
  const { error } = await withSubmitTimeout(query, "删除标注提交");
  if (error) throw error;
}

async function deleteHumanAnnotationById(id) {
  const { error } = await withSubmitTimeout(client.from("human_annotations").delete().eq("id", id), "删除标注提交");
  if (error) throw error;
}

async function deleteHumanAnnotationsBySong(songId) {
  const { error } = await withSubmitTimeout(client.from("human_annotations").delete().eq("song_id", songId), "删除标注提交");
  if (error) throw error;
}

async function deleteReviewsByAutoSection(autoSectionId, songId = state.selectedSong?.id) {
  let query = client.from("section_reviews").delete().eq("auto_section_id", autoSectionId);
  if (songId) query = query.eq("song_id", songId);
  const { error } = await withSubmitTimeout(query, "删除质检记录提交");
  if (error) throw error;
}

async function deleteReviewsByAutoSectionIds(autoSectionIds, songId = state.selectedSong?.id) {
  if (!autoSectionIds.length) return;
  let query = client.from("section_reviews").delete().in("auto_section_id", autoSectionIds);
  if (songId) query = query.eq("song_id", songId);
  const { error } = await withSubmitTimeout(query, "删除质检记录提交");
  if (error) throw error;
}

async function deleteReviewsByHumanAnnotationId(annotationId) {
  const { error } = await withSubmitTimeout(
    client.from("section_reviews").delete().eq("human_annotation_id", annotationId),
    "删除质检记录提交",
  );
  if (error) throw error;
}

async function deleteReviewsByHumanAnnotationIds(annotationIds) {
  if (!annotationIds.length) return;
  const { error } = await withSubmitTimeout(
    client.from("section_reviews").delete().in("human_annotation_id", annotationIds),
    "删除质检记录提交",
  );
  if (error) throw error;
}

async function deleteReviewsBySong(songId) {
  const { error } = await withSubmitTimeout(client.from("section_reviews").delete().eq("song_id", songId), "删除质检记录提交");
  if (error) throw error;
}

async function updateSongStatus(status, songId = state.selectedSong?.id) {
  if (!songId) throw new Error("缺少歌曲 id，无法更新歌曲状态");
  const { error } = await withSubmitTimeout(client.from("songs").update({ status }).eq("id", songId), "歌曲状态提交");
  if (error) throw error;
  if (state.selectedSong?.id === songId) state.selectedSong.status = status;
  const summarySong = state.songs.find((song) => song.id === songId);
  if (summarySong) {
    summarySong.status = status;
    summarySong.is_reviewed = status === "reviewed";
  }
  renderSummary();
  renderSongPicker();
}

function restoreModelValues() {
  if (!state.selectedSection) return;
  const section = state.selectedSection;
  rememberEditorState("恢复机器");
  els.editType.value = section.section_type;
  els.editStart.value = displayTime(section.start_time_sec);
  els.editEnd.value = displayTime(section.end_time_sec);
  renderTypeButtonState();
  showToast("已恢复机器值，可用 ⌘Z 撤销");
}

function selectRelativeSection(delta) {
  if (!state.selectedSection) return;
  const sections = visibleSections();
  const index = sections.findIndex((section) => section.id === state.selectedSection.id);
  const next = sections[index + delta];
  if (next) previewSection(next.id);
}

function localSectionMutationGuard() {
  if (state.isSubmittingSong || state.submittingSectionIds.size) {
    showToast("正在提交中，不能修改 section", true);
    return false;
  }
  return true;
}

async function addSectionAfterCurrent() {
  if (!state.selectedSong || !state.run) return showToast("先选择一首歌", true);
  if (!localSectionMutationGuard()) return;
  const sections = visibleSections();
  const current = state.selectedSection || sections.at(-1);
  const source = current ? finalResult(current) : null;
  const start = source ? Number(source.start_time_sec) : 0;
  const end = source ? Number(source.end_time_sec) : Math.min(start + 12, Number(state.selectedSong.duration_sec || start + 12));
  const nextLocalIndex = state.sections.filter((section) => section.isLocal).length + 1;
  const previousSelectedSectionId = state.selectedSection?.id || null;
  const section = {
    id: `local-${Date.now()}`,
    isLocal: true,
    local_label: `新增 Section ${nextLocalIndex}`,
    section_index: current ? Number(current.section_index || activeSectionIndex(current.id) + 1) + 1 : activeSections().length + 1,
    start_time_sec: start,
    end_time_sec: Math.max(start + 1, end),
    section_type: source?.section_type || "verse",
    lyric_evidence: "人工新增",
    vocal_evidence: current?.vocal_evidence || "",
    acoustic_evidence: current?.acoustic_evidence || "",
    need_human_review: true,
  };
  insertSectionAfter(section, current?.id || null);
  renumberSections();
  try {
    const context = createSubmitContext(activeSections());
    const values = {
      type: section.section_type,
      start: Number(section.start_time_sec),
      end: Number(section.end_time_sec),
    };
    const payload = buildPayload(
      section,
      {
        ...values,
        confidence: 1,
        comment: section.local_label,
        qcStatus: "human_modified",
        action: "edit",
      },
      context,
    );
    showToast("新增 section 正在写入数据库", false, { persistent: true });
    const annotation = await upsertAnnotation(section, payload, "edit", context);
    await verifyAnnotationWrite(annotation.id, {
      songId: context.songId,
      runId: context.runId,
      autoSectionId: null,
    }, annotation._review_id);
    await verifyReadableAnnotation(annotation.id);
    await updateSongStatus("in_review", context.songId);
    markSectionSubmitted(section);
    pushUndoEntry({
      label: "新增 section",
      type: "remove-local-section",
      sectionId: section.id,
      previousSelectedSectionId,
      insertIndex: activeSectionIndex(section.id),
    });
    selectSection(section.id);
    flashSectionEffect(section.id, "section-entering", 360);
    showToast("新增 section 已写入数据库");
  } catch (error) {
    state.sections = state.sections.filter((item) => item.id !== section.id);
    renumberSections();
    if (previousSelectedSectionId) selectSection(previousSelectedSectionId);
    else renderSections();
    showToast(submitErrorMessage(error, "新增 section 写入失败"), true);
  }
}

async function deleteCurrentSection() {
  const section = state.selectedSection;
  if (!section) return showToast("先选择一个 section", true);
  if (!localSectionMutationGuard()) return;
  const sectionIndex = visibleSections().findIndex((item) => item.id === section.id);
  const context = createSubmitContext(activeSections());
  let fallbackSection = null;
  try {
    showToast("删除 section 正在写入数据库", false, { persistent: true });
    const boundarySync = await connectAroundDeletedSection(section, context);
    fallbackSection = boundarySync?.section || null;
    if (boundarySync?.annotation) {
      await verifyAnnotationWrite(boundarySync.annotation.id, {
        songId: context.songId,
        runId: context.runId,
        autoSectionId: fallbackSection?.isLocal ? null : fallbackSection?.id,
      }, boundarySync.annotation._review_id);
    }
    if (section.isLocal) {
      if (section.saved_annotation_id) {
        await deleteReviewsByHumanAnnotationId(section.saved_annotation_id);
        await deleteHumanAnnotationById(section.saved_annotation_id);
      }
    } else {
      const annotationIds = humanAnnotationIdsForSection(section);
      await deleteReviewsByHumanAnnotationIds(annotationIds);
      await deleteHumanAnnotationsByAutoSection(section.id, context.songId);
      await recordDeletedSectionReviews([section.id], context);
      await verifyDeletedSectionReview(context, section.id);
    }
    await updateSongStatus("in_review", context.songId);
  } catch (error) {
    showToast(submitErrorMessage(error, "删除 section 写入失败"), true);
    return;
  }
  const removeSection = () => {
    if (section.isLocal) {
      state.sections = state.sections.filter((item) => item.id !== section.id);
    } else {
      state.deletedSectionIds.add(section.id);
      persistDeletedSectionIds();
    }
    renumberSections();
    const next =
      (fallbackSection && visibleSections().find((item) => item.id === fallbackSection.id)) ||
      visibleSections()[Math.max(0, Math.min(sectionIndex, visibleSections().length - 1))];
    if (next) selectSection(next.id);
    else {
      clearInspector();
      renderSections();
    }
    showToast("删除 section 已写入数据库");
  };
  removeSectionAfterExit(section.id, removeSection);
}

function startSectionDrag(event, sectionId) {
  if (!event.target.closest(".section-drag-handle")) {
    event.preventDefault();
    return;
  }
  state.draggedSectionId = sectionId;
  event.dataTransfer.effectAllowed = "move";
  event.dataTransfer.setData("text/plain", sectionId);
  event.currentTarget.classList.add("dragging");
}

function moveSectionDragOver(event, targetSectionId) {
  if (!state.draggedSectionId || state.draggedSectionId === targetSectionId) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = "move";
  for (const row of els.sectionList.querySelectorAll(".section-row.drop-target")) {
    row.classList.remove("drop-target", "drop-before", "drop-after");
  }
  event.currentTarget.classList.add("drop-target", sectionDropPosition(event, event.currentTarget) === "after" ? "drop-after" : "drop-before");
}

function dropSectionOn(event, targetSectionId) {
  if (!state.draggedSectionId || state.draggedSectionId === targetSectionId) return;
  event.preventDefault();
  const moved = moveSectionRelativeTo(state.draggedSectionId, targetSectionId, sectionDropPosition(event, event.currentTarget));
  const draggedSectionId = state.draggedSectionId;
  endSectionDrag();
  if (moved) {
    preserveSectionListScroll(() => selectSectionById(draggedSectionId, { scroll: false }));
    showToast("已调整 section 顺序，未写入数据库，请提交整首歌");
  }
}

function endSectionDrag() {
  state.draggedSectionId = null;
  for (const row of els.sectionList.querySelectorAll(".section-row.dragging, .section-row.drop-target")) {
    row.classList.remove("dragging", "drop-target", "drop-before", "drop-after");
  }
}

function startSectionPointerDrag(event, sectionId) {
  if (event.button !== 0) return;
  event.preventDefault();
  event.stopPropagation();
  const row = event.currentTarget.closest(".section-row");
  state.pointerSectionDrag = {
    sectionId,
    pointerId: event.pointerId,
    startY: event.clientY,
    targetId: null,
    position: "before",
    moved: false,
  };
  state.draggedSectionId = sectionId;
  row?.classList.add("dragging");
  event.currentTarget.setPointerCapture?.(event.pointerId);
  window.addEventListener("pointermove", moveSectionPointerDrag);
  window.addEventListener("pointerup", endSectionPointerDrag);
  window.addEventListener("pointercancel", cancelSectionPointerDrag);
}

function moveSectionPointerDrag(event) {
  const drag = state.pointerSectionDrag;
  if (!drag || event.pointerId !== drag.pointerId) return;
  event.preventDefault();
  drag.moved = drag.moved || Math.abs(event.clientY - drag.startY) > 4;
  const targetRow = sectionRowAtPoint(event.clientX, event.clientY);
  clearSectionDropIndicators();
  if (!targetRow || targetRow.dataset.sectionId === drag.sectionId) return;
  drag.targetId = targetRow.dataset.sectionId;
  drag.position = sectionDropPosition({ clientY: event.clientY }, targetRow);
  targetRow.classList.add("drop-target", drag.position === "after" ? "drop-after" : "drop-before");
}

function endSectionPointerDrag(event) {
  const drag = state.pointerSectionDrag;
  if (!drag || event.pointerId !== drag.pointerId) return;
  const moved = drag.moved && drag.targetId ? moveSectionRelativeTo(drag.sectionId, drag.targetId, drag.position) : false;
  const draggedSectionId = drag.sectionId;
  cancelSectionPointerDrag();
  if (moved) {
    preserveSectionListScroll(() => selectSectionById(draggedSectionId, { scroll: false }));
    showToast("已调整 section 顺序，未写入数据库，请提交整首歌");
  }
}

function cancelSectionPointerDrag() {
  state.pointerSectionDrag = null;
  state.draggedSectionId = null;
  clearSectionDropIndicators();
  window.removeEventListener("pointermove", moveSectionPointerDrag);
  window.removeEventListener("pointerup", endSectionPointerDrag);
  window.removeEventListener("pointercancel", cancelSectionPointerDrag);
}

function sectionRowAtPoint(x, y) {
  return document
    .elementsFromPoint(x, y)
    .map((element) => element.closest?.(".section-row"))
    .find(Boolean) || null;
}

function clearSectionDropIndicators() {
  for (const row of els.sectionList.querySelectorAll(".section-row.dragging, .section-row.drop-target")) {
    row.classList.remove("dragging", "drop-target", "drop-before", "drop-after");
  }
  if (state.draggedSectionId) {
    [...els.sectionList.querySelectorAll(".section-row")]
      .find((row) => row.dataset.sectionId === String(state.draggedSectionId))
      ?.classList.add("dragging");
  }
}

function sectionDropPosition(event, row) {
  const rect = row.getBoundingClientRect();
  return event.clientY > rect.top + rect.height / 2 ? "after" : "before";
}

function renumberSections() {
  activeSections().forEach((section, index) => {
    section.section_index = index + 1;
  });
}

async function deleteDeletedAnnotations(context = null) {
  if (context) assertSubmitContext(context);
  const songId = submitContextSongId(context);
  const ids = [...state.deletedSectionIds].filter((id) => !String(id).startsWith("local-"));
  if (context) {
    for (const id of ids) {
      const section = state.sections.find((item) => item.id === id);
      if (section?.run_id && section.run_id !== context.runId) {
        throw new Error("删除队列里存在不属于当前 model_run 的 section，已阻止写入数据库");
      }
    }
  }
  if (ids.length) {
    const annotationIds = state.annotations
      .filter((annotation) => annotation.song_id === songId && ids.includes(annotation.source_auto_section_id))
      .map((annotation) => annotation.id)
      .filter(Boolean);
    await deleteReviewsByAutoSectionIds(ids, songId);
    await deleteReviewsByHumanAnnotationIds(annotationIds);
    const { error } = await withSubmitTimeout(
      client.from("human_annotations").delete().eq("song_id", songId).in("source_auto_section_id", ids),
      "删除标注提交",
    );
    if (error) throw error;
    const alreadyDeleted = deletedSectionIdsFromReviews(state.reviews);
    await recordDeletedSectionReviews(
      ids.filter((id) => !alreadyDeleted.has(id)),
      context,
    );
  }
  const localAnnotationIds = [...state.deletedLocalAnnotationIds];
  if (localAnnotationIds.length) {
    await deleteReviewsByHumanAnnotationIds(localAnnotationIds);
    const { error } = await withSubmitTimeout(
      client.from("human_annotations").delete().eq("song_id", songId).in("id", localAnnotationIds),
      "删除标注提交",
    );
    if (error) throw error;
  }
}

async function recordDeletedSectionReviews(autoSectionIds, context) {
  if (!autoSectionIds.length) return;
  const songId = submitContextSongId(context);
  const runId = submitContextRunId(context);
  const rows = autoSectionIds.map((autoSectionId) => ({
    song_id: songId,
    run_id: runId,
    auto_section_id: autoSectionId,
    human_annotation_id: null,
    action: "edit",
    reviewer: "local_reviewer",
    before_json: state.sections.find((section) => section.id === autoSectionId) || null,
    after_json: { deleted: true },
  }));
  const { data, error } = await withSubmitTimeout(client.from("section_reviews").insert(rows).select(), "删除记录提交");
  if (error) throw error;
  state.reviews = [...(data || []), ...state.reviews];
}

function seekFromTimeline(event, total, shouldPlay = false) {
  if (state.timelineDrag?.moved) return;
  const rect = els.timeline.getBoundingClientRect();
  if (!rect.width || !Number.isFinite(total)) return;
  const x = event.clientX - rect.left + els.timeline.scrollLeft;
  const ratio = Math.max(0, Math.min(1, x / (rect.width * state.zoom)));
  locatePlaybackAt(ratio * total, { play: shouldPlay, scrollSection: true });
}

function updateTimelinePlayhead() {
  const playhead = document.getElementById("timelinePlayhead");
  const total = sectionTotalDuration();
  const ratio = total > 0 ? Math.max(0, Math.min(1, Number(els.audioPlayer.currentTime || 0) / total)) : 0;
  if (playhead) playhead.style.left = `${ratio * 100}%`;
  updateTransport();
  updateActiveLyric();
  updateActiveSectionFromPlayback();
  if (!state.isScrubbing && state.previewEndTime !== null && Number(els.audioPlayer.currentTime || 0) >= state.previewEndTime) {
    els.audioPlayer.pause();
    state.previewEndTime = null;
  }
}

function updateActiveSectionFromPlayback() {
  if (state.syncingSectionFromPlayback || !state.sections.length) return;
  const current = Number(els.audioPlayer.currentTime || 0);
  if (state.selectedSection && sectionContainsTime(state.selectedSection, current)) return;
  const section = sectionForTime(current);
  if (!section || state.selectedSection?.id === section.id) return;
  state.syncingSectionFromPlayback = true;
  state.selectedSection = section;
  fillInspector(section);
  renderSections();
  scrollActiveSectionIntoView();
  state.syncingSectionFromPlayback = false;
}

function sectionForTime(current) {
  return state.sections
    .filter((item) => !state.deletedSectionIds.has(item.id))
    .find((item) => sectionContainsTime(item, current));
}

function sectionContainsTime(section, current) {
  const final = finalResult(section);
  return current >= timeToSeconds(final.start_time_sec, 0) && current < timeToSeconds(final.end_time_sec, 0);
}

function sectionTotalDuration() {
  const sectionEnds = state.sections
    .filter((section) => !state.deletedSectionIds.has(section.id))
    .map((section) => timeToSeconds(finalResult(section).end_time_sec, 0));
  return Math.max(Number(els.audioPlayer.duration || 0), Number(state.selectedSong?.duration_sec || 0), ...sectionEnds, 0);
}

function updateTransport() {
  const duration = sectionTotalDuration();
  const current = Number(els.audioPlayer.currentTime || 0);
  els.progressRange.max = duration || 0;
  els.progressRange.value = Math.min(current, duration || current);
  els.currentTimeText.textContent = formatTime(current);
  els.durationText.textContent = formatTime(duration);
  els.playToggle.textContent = els.audioPlayer.paused ? "▶" : "Ⅱ";
  if (els.lyricsModalPlay) els.lyricsModalPlay.textContent = els.audioPlayer.paused ? "▶" : "Ⅱ";
  els.bottomProgressRange.max = duration || 0;
  els.bottomProgressRange.value = Math.min(current, duration || current);
  els.bottomCurrentTimeText.textContent = formatTime(current);
  els.bottomDurationText.textContent = formatTime(duration);
  els.bottomPlayToggle.textContent = els.audioPlayer.paused ? "▶" : "Ⅱ";
  updateSpeedButtons();
}

function updateSpeedButtons() {
  const label = `${Number(els.audioPlayer.playbackRate || state.playbackRate).toString()}x`;
  if (els.speedToggle) els.speedToggle.textContent = label;
  if (els.bottomSpeedToggle) els.bottomSpeedToggle.textContent = label;
}

function setPlaybackRate(rate) {
  state.playbackRate = rate;
  els.audioPlayer.playbackRate = rate;
  updateSpeedButtons();
}

function cyclePlaybackRate() {
  const current = Number(els.audioPlayer.playbackRate || state.playbackRate || 1);
  const index = playbackRates.findIndex((rate) => Math.abs(rate - current) < 0.01);
  setPlaybackRate(playbackRates[(index + 1) % playbackRates.length]);
}

function renderLyricsViewport() {
  state.currentLyricIndex = null;
  if (!state.lyrics.length) {
    els.lyricsViewport.textContent = "没有歌词数据";
    return;
  }
  els.lyricsViewport.innerHTML = state.lyrics
    .map(
      (line, index) => `
        <button type="button" class="lyric-scroll-line" data-lyric-index="${index}" data-start="${Number(line.start_time_sec || 0)}">
          <time>${displayTime(line.start_time_sec, line.start_time_label)}</time>
          <span>${escapeHtml(toSimplified(line.text))}</span>
        </button>
      `,
    )
    .join("");
  for (const line of els.lyricsViewport.querySelectorAll(".lyric-scroll-line")) {
    line.addEventListener("click", (event) => {
      if (event.detail > 1 || line.classList.contains("editing")) return;
      locatePlaybackAt(line.dataset.start, { play: true, scrollSection: true });
    });
    line.addEventListener("dblclick", (event) => {
      event.preventDefault();
      event.stopPropagation();
      beginLyricEdit(Number(line.dataset.lyricIndex));
    });
  }
}

function beginLyricEdit(index) {
  const line = state.lyrics[index];
  const node = els.lyricsViewport.querySelector(`[data-lyric-index="${index}"]`);
  if (!line || !node || node.classList.contains("editing")) return;
  const originalText = toSimplified(line.text);
  node.classList.add("editing");
  node.innerHTML = `
    <time>${displayTime(line.start_time_sec, line.start_time_label)}</time>
    <input class="lyric-edit-input" value="${escapeHtml(originalText)}" />
  `;
  const input = node.querySelector(".lyric-edit-input");
  input.focus();
  input.select();

  let isSaving = false;
  const save = async () => {
    if (isSaving) return;
    isSaving = true;
    const nextText = input.value.trim();
    if (!nextText) {
      renderLyricsViewport();
      updateActiveLyric();
      return showToast("歌词不能为空", true);
    }
    if (nextText === originalText) {
      renderLyricsViewport();
      updateActiveLyric();
      return;
    }
    try {
      showToast("歌词正在提交", false, { persistent: true });
      await updateLyricLine(line, nextText);
      line.text = nextText;
      renderLyricsViewport();
      updateActiveLyric();
      showToast("歌词提交成功");
    } catch (error) {
      isSaving = false;
      showToast(submitErrorMessage(error, "歌词提交失败"), true);
    }
  };

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") input.blur();
    if (event.key === "Escape") {
      event.preventDefault();
      renderLyricsViewport();
      updateActiveLyric();
    }
  });
  input.addEventListener("blur", save);
}

async function updateLyricLine(line, text) {
  const query = client.from("lyric_lines").update({ text });
  const request = line.id
    ? query.eq("id", line.id)
    : query.eq("song_id", state.selectedSong.id).eq("line_index", line.line_index);
  const { error } = await withSubmitTimeout(request, "歌词提交");
  if (error) throw error;
}

function openLyricsModal() {
  if (!state.selectedSong) return showToast("先选择一首歌", true);
  if (!state.lyrics.length) {
    const start = Number(els.audioPlayer.currentTime || 0);
    state.currentLyricIndex = 0;
    renderLyricBulkRows([{ start_time_sec: start, end_time_sec: start + 4, text: "" }]);
  } else {
    updateActiveLyric({ scroll: false });
    renderLyricBulkRows();
  }
  state.lyricUndoStack = [];
  state.lastLyricSnapshotKey = lyricSnapshotKey(lyricSnapshot());
  els.lyricsModal.classList.add("open");
  els.lyricsModal.setAttribute("aria-hidden", "false");
  ensureLyricModalFrame();
  requestAnimationFrame(() => scrollLyricBulkActiveIntoView({ focus: true }));
}

function closeLyricsModal() {
  els.lyricsModal.classList.remove("open");
  els.lyricsModal.setAttribute("aria-hidden", "true");
}

function ensureLyricModalFrame() {
  if (!els.lyricsModalPanel) return;
  if (!state.lyricModalFrame) {
    const width = Math.min(920, Math.max(720, window.innerWidth * 0.72));
    const height = Math.min(720, Math.max(460, width * 0.72));
    state.lyricModalFrame = {
      width,
      height,
      left: Math.max(16, (window.innerWidth - width) / 2),
      top: Math.max(16, (window.innerHeight - height) / 2),
    };
  }
  applyLyricModalFrame();
}

function applyLyricModalFrame(ratio = null) {
  const frame = constrainLyricModalFrame(state.lyricModalFrame, ratio);
  state.lyricModalFrame = frame;
  Object.assign(els.lyricsModalPanel.style, {
    width: `${frame.width}px`,
    height: `${frame.height}px`,
    left: `${frame.left}px`,
    top: `${frame.top}px`,
  });
}

function constrainLyricModalFrame(frame, ratio = null) {
  const margin = 12;
  const minWidth = Math.min(720, window.innerWidth - margin * 2);
  const minHeight = Math.min(460, window.innerHeight - margin * 2);
  const maxWidth = Math.max(minWidth, window.innerWidth - margin * 2);
  const maxHeight = Math.max(minHeight, window.innerHeight - margin * 2);
  let width = Math.max(minWidth, Math.min(maxWidth, Number(frame.width || minWidth)));
  let height = Math.max(minHeight, Math.min(maxHeight, Number(frame.height || minHeight)));
  if (ratio && Number.isFinite(ratio) && ratio > 0) {
    width = Math.max(minWidth, Math.min(maxWidth, width));
    height = width * ratio;
    if (height > maxHeight) {
      height = maxHeight;
      width = height / ratio;
    }
    if (height < minHeight) {
      height = minHeight;
      width = height / ratio;
    }
    width = Math.max(minWidth, Math.min(maxWidth, width));
    height = width * ratio;
  }
  const left = Math.max(margin, Math.min(window.innerWidth - width - margin, Number(frame.left || margin)));
  const top = Math.max(margin, Math.min(window.innerHeight - height - margin, Number(frame.top || margin)));
  return { width, height, left, top };
}

function startLyricModalMove(event) {
  if (event.target.closest("button")) return;
  ensureLyricModalFrame();
  event.preventDefault();
  state.lyricModalDrag = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    frame: { ...state.lyricModalFrame },
  };
  event.currentTarget.setPointerCapture?.(event.pointerId);
  window.addEventListener("pointermove", moveLyricModal);
  window.addEventListener("pointerup", endLyricModalMove);
  window.addEventListener("pointercancel", endLyricModalMove);
}

function moveLyricModal(event) {
  const drag = state.lyricModalDrag;
  if (!drag || event.pointerId !== drag.pointerId) return;
  state.lyricModalFrame = {
    ...drag.frame,
    left: drag.frame.left + event.clientX - drag.startX,
    top: drag.frame.top + event.clientY - drag.startY,
  };
  applyLyricModalFrame();
}

function endLyricModalMove(event) {
  if (state.lyricModalDrag && event.pointerId !== state.lyricModalDrag.pointerId) return;
  state.lyricModalDrag = null;
  window.removeEventListener("pointermove", moveLyricModal);
  window.removeEventListener("pointerup", endLyricModalMove);
  window.removeEventListener("pointercancel", endLyricModalMove);
}

function startLyricModalResize(event) {
  ensureLyricModalFrame();
  event.preventDefault();
  event.stopPropagation();
  state.lyricModalResize = {
    pointerId: event.pointerId,
    startX: event.clientX,
    frame: { ...state.lyricModalFrame },
    ratio: state.lyricModalFrame.height / state.lyricModalFrame.width,
  };
  event.currentTarget.setPointerCapture?.(event.pointerId);
  window.addEventListener("pointermove", resizeLyricModal);
  window.addEventListener("pointerup", endLyricModalResize);
  window.addEventListener("pointercancel", endLyricModalResize);
}

function resizeLyricModal(event) {
  const drag = state.lyricModalResize;
  if (!drag || event.pointerId !== drag.pointerId) return;
  const nextWidth = drag.frame.width + event.clientX - drag.startX;
  state.lyricModalFrame = {
    ...drag.frame,
    width: nextWidth,
    height: nextWidth * drag.ratio,
  };
  applyLyricModalFrame(drag.ratio);
}

function endLyricModalResize(event) {
  if (state.lyricModalResize && event.pointerId !== state.lyricModalResize.pointerId) return;
  state.lyricModalResize = null;
  window.removeEventListener("pointermove", resizeLyricModal);
  window.removeEventListener("pointerup", endLyricModalResize);
  window.removeEventListener("pointercancel", endLyricModalResize);
}

function renderLyricBulkRows(rows = state.lyrics) {
  if (!els.lyricsBulkList) return;
  els.lyricsBulkList.innerHTML = rows
    .map(
      (line, index) => `
        <div class="lyric-bulk-row ${index === state.currentLyricIndex ? "active" : ""}" data-lyric-index="${index}">
          <span class="lyric-bulk-index">${index + 1}</span>
          <input class="lyric-bulk-start" type="text" inputmode="numeric" value="${displayTime(line.start_time_sec, line.start_time_label)}" aria-label="歌词开始时间" />
          <input class="lyric-bulk-end" type="text" inputmode="numeric" value="${displayTime(line.end_time_sec, line.end_time_label)}" aria-label="歌词结束时间" />
          <input class="lyric-bulk-text" type="text" value="${escapeHtml(toSimplified(line.text))}" aria-label="歌词文本" />
          <button type="button" class="lyric-bulk-delete" aria-label="删除歌词行">删除</button>
        </div>
      `,
    )
    .join("");
  for (const button of els.lyricsBulkList.querySelectorAll(".lyric-bulk-delete")) {
    button.addEventListener("click", () => {
      rememberLyricState("删除歌词行");
      const row = button.closest(".lyric-bulk-row");
      const deletedIndex = Number(row?.dataset.lyricIndex || 0);
      row?.remove();
      renumberLyricBulkRows();
      const nextCount = els.lyricsBulkList.querySelectorAll(".lyric-bulk-row").length;
      state.currentLyricIndex = nextCount ? Math.min(deletedIndex, nextCount - 1) : null;
      updateLyricModalActive({ scroll: false });
    });
  }
  for (const row of els.lyricsBulkList.querySelectorAll(".lyric-bulk-row")) {
    row.addEventListener("click", (event) => {
      if (event.target.closest("input, button")) return;
      seekLyricBulkRow(row);
    });
    row.addEventListener("focusin", () => {
      state.currentLyricIndex = Number(row.dataset.lyricIndex);
      updateLyricModalActive({ scroll: false });
    });
    for (const input of row.querySelectorAll("input")) {
      input.addEventListener("focus", () => rememberLyricState("编辑歌词"));
      input.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        syncAdjacentLyricTime(input);
        focusNextLyricBulkInput(input);
      });
      if (input.classList.contains("lyric-bulk-start") || input.classList.contains("lyric-bulk-end")) {
        input.addEventListener("input", () => syncAdjacentLyricTime(input));
      }
      input.addEventListener("change", () => syncAdjacentLyricTime(input));
    }
  }
}

function lyricSnapshot() {
  return {
    rows: readLyricBulkRows({ allowEmptyText: true, skipValidation: true }).map((row) => ({ ...row })),
    currentLyricIndex: state.currentLyricIndex,
  };
}

function lyricSnapshotKey(snapshot) {
  return JSON.stringify(snapshot?.rows || []);
}

function rememberLyricState(label = "编辑歌词") {
  if (!els.lyricsModal.classList.contains("open") || !els.lyricsBulkList) return;
  const snapshot = lyricSnapshot();
  const key = lyricSnapshotKey(snapshot);
  if (key === state.lastLyricSnapshotKey) return;
  state.lyricUndoStack.push({ label, snapshot });
  if (state.lyricUndoStack.length > 10) state.lyricUndoStack.shift();
  state.lastLyricSnapshotKey = key;
}

function restoreLyricSnapshot(snapshot) {
  if (!snapshot?.rows?.length) return false;
  state.currentLyricIndex = snapshot.currentLyricIndex;
  renderLyricBulkRows(snapshot.rows);
  updateLyricModalActive({ scroll: false });
  state.lastLyricSnapshotKey = lyricSnapshotKey(snapshot);
  return true;
}

function undoLastLyricAction() {
  while (state.lyricUndoStack.length) {
    const entry = state.lyricUndoStack.pop();
    if (restoreLyricSnapshot(entry.snapshot)) {
      showToast(`已撤销：${entry.label}`);
      return;
    }
  }
  showToast("没有可撤销的歌词操作");
}

function renumberLyricBulkRows() {
  if (!els.lyricsBulkList) return;
  [...els.lyricsBulkList.querySelectorAll(".lyric-bulk-row")].forEach((row, index) => {
    row.dataset.lyricIndex = String(index);
    const label = row.querySelector(".lyric-bulk-index");
    if (label) label.textContent = String(index + 1);
  });
}

function addLyricBulkRow() {
  rememberLyricState("新增歌词行");
  const rows = readLyricBulkRows({ allowEmptyText: true, skipValidation: true });
  const currentIndex = currentLyricBulkIndex(rows.length);
  const current = rows[currentIndex] || rows.at(-1);
  const next = rows[currentIndex + 1];
  const start = current ? current.end_time_sec : Number(els.audioPlayer.currentTime || 0);
  let end = next ? next.start_time_sec : start + 4;
  if (end <= start) end = start + 4;
  const insertIndex = Math.min(rows.length, currentIndex + 1);
  rows.splice(insertIndex, 0, { start_time_sec: start, end_time_sec: end, text: "" });
  state.currentLyricIndex = insertIndex;
  renderLyricBulkRows(rows);
  requestAnimationFrame(() => {
    const textInput = els.lyricsBulkList?.querySelector(`[data-lyric-index="${insertIndex}"] .lyric-bulk-text`);
    textInput?.focus();
  });
}

function currentLyricBulkIndex(rowCount) {
  const focusedRow = document.activeElement?.closest?.(".lyric-bulk-row");
  const focusedIndex = Number(focusedRow?.dataset.lyricIndex);
  if (Number.isInteger(focusedIndex) && focusedIndex >= 0 && focusedIndex < rowCount) return focusedIndex;
  const activeIndex = Number(state.currentLyricIndex);
  if (Number.isInteger(activeIndex) && activeIndex >= 0 && activeIndex < rowCount) return activeIndex;
  return Math.max(0, rowCount - 1);
}

function readLyricBulkRows({ allowEmptyText = false, skipValidation = false } = {}) {
  const rows = [...(els.lyricsBulkList?.querySelectorAll(".lyric-bulk-row") || [])];
  return rows.map((row, index) => {
    const startText = row.querySelector(".lyric-bulk-start")?.value;
    const endText = row.querySelector(".lyric-bulk-end")?.value;
    const lyricText = row.querySelector(".lyric-bulk-text")?.value.trim() || "";
    const start = skipValidation ? timeToSeconds(startText, 0) : parseTimeInput(startText);
    const end = skipValidation ? timeToSeconds(endText, start + 4) : parseTimeInput(endText);
    if (!skipValidation && end <= start) throw new Error(`第 ${index + 1} 行结束时间必须晚于开始时间`);
    if (!allowEmptyText && !lyricText) throw new Error(`第 ${index + 1} 行歌词不能为空`);
    return { start_time_sec: start, end_time_sec: end, text: lyricText };
  });
}

function syncAdjacentLyricTime(input) {
  const row = input.closest(".lyric-bulk-row");
  if (!row) return;
  try {
    parseTimeInput(input.value);
  } catch {
    return;
  }
  const index = Number(row.dataset.lyricIndex);
  if (input.classList.contains("lyric-bulk-start")) {
    const previous = els.lyricsBulkList?.querySelector(`[data-lyric-index="${index - 1}"] .lyric-bulk-end`);
    if (previous && previous.value !== input.value) {
      rememberLyricState("联动歌词时间");
      previous.value = input.value;
      flashLyricLinkedInput(previous);
    }
  }
  if (input.classList.contains("lyric-bulk-end")) {
    const next = els.lyricsBulkList?.querySelector(`[data-lyric-index="${index + 1}"] .lyric-bulk-start`);
    if (next && next.value !== input.value) {
      rememberLyricState("联动歌词时间");
      next.value = input.value;
      flashLyricLinkedInput(next);
    }
  }
}

function flashLyricLinkedInput(input) {
  input.classList.remove("lyric-linked-flash");
  void input.offsetWidth;
  input.classList.add("lyric-linked-flash");
}

function focusNextLyricBulkInput(input) {
  const row = input.closest(".lyric-bulk-row");
  if (!row) return;
  const fieldClass = ["lyric-bulk-start", "lyric-bulk-end", "lyric-bulk-text"].find((name) => input.classList.contains(name));
  const currentIndex = Number(row.dataset.lyricIndex);
  let nextRow = els.lyricsBulkList?.querySelector(`[data-lyric-index="${currentIndex + 1}"]`);
  if (!nextRow) {
    addLyricBulkRow();
    requestAnimationFrame(() => {
      const created = els.lyricsBulkList?.querySelector(`[data-lyric-index="${currentIndex + 1}"]`);
      const target = created?.querySelector(fieldClass ? `.${fieldClass}` : ".lyric-bulk-text");
      target?.focus();
      target?.select?.();
    });
    return;
  }
  state.currentLyricIndex = currentIndex + 1;
  updateLyricModalActive({ scroll: false });
  const target = nextRow.querySelector(fieldClass ? `.${fieldClass}` : ".lyric-bulk-text");
  target?.focus();
  target?.select?.();
}

function seekLyricBulkRow(row) {
  try {
    const start = parseTimeInput(row.querySelector(".lyric-bulk-start")?.value);
    state.currentLyricIndex = Number(row.dataset.lyricIndex);
    updateLyricModalActive({ scroll: false });
    locatePlaybackAt(start, { play: true, scrollSection: true });
    state.currentLyricIndex = Number(row.dataset.lyricIndex);
    updateLyricModalActive({ scroll: false });
    updateCompactLyricActiveState(state.currentLyricIndex, false);
  } catch (error) {
    showToast(error.message, true);
  }
}

function scrollLyricBulkActiveIntoView({ focus = false } = {}) {
  if (!els.lyricsBulkList) return;
  const active =
    els.lyricsBulkList.querySelector(".lyric-bulk-row.active") ||
    els.lyricsBulkList.querySelector(`[data-lyric-index="${state.currentLyricIndex}"]`);
  if (!active) return;
  active.scrollIntoView({ block: "center" });
  if (focus) active.querySelector(".lyric-bulk-text")?.focus();
}

function updateLyricModalActive({ scroll = false } = {}) {
  if (!els.lyricsModal?.classList.contains("open") || !els.lyricsBulkList) return;
  for (const row of els.lyricsBulkList.querySelectorAll(".lyric-bulk-row")) {
    row.classList.toggle("active", Number(row.dataset.lyricIndex) === state.currentLyricIndex);
  }
  if (scroll) scrollLyricBulkActiveIntoView();
}

async function saveBulkLyrics() {
  if (!state.selectedSong) return showToast("先选择一首歌", true);
  els.lyricsBulkSave.disabled = true;
  els.lyricsBulkSave.textContent = "提交中...";
  try {
    const rows = readLyricBulkRows();
    if (!rows.length) throw new Error("至少保留一行歌词");
    showToast("歌词正在提交", false, { persistent: true });
    await replaceSongLyrics(rows);
    state.lyricUndoStack = [];
    state.lastLyricSnapshotKey = "";
    renderLyricsViewport();
    updateActiveLyric();
    closeLyricsModal();
    showToast(`歌词提交成功，当前 ${state.lyrics.length} 行`);
  } catch (error) {
    showToast(submitErrorMessage(error, "歌词提交失败"), true);
  } finally {
    els.lyricsBulkSave.disabled = false;
    els.lyricsBulkSave.textContent = "保存歌词";
  }
}

async function replaceSongLyrics(rows) {
  const source = state.lyrics.find((line) => line.source)?.source || "manual";
  const payload = rows.map((row, index) => ({
    song_id: state.selectedSong.id,
    line_index: index,
    start_time_sec: row.start_time_sec,
    end_time_sec: row.end_time_sec,
    text: row.text,
    source,
  }));
  const upsertResponse = await withSubmitTimeout(
    client.from("lyric_lines").upsert(payload, { onConflict: "song_id,line_index" }).select(),
    "歌词提交",
  );
  if (upsertResponse.error) throw upsertResponse.error;
  const deleteTailResponse = await withSubmitTimeout(
    client.from("lyric_lines").delete().eq("song_id", state.selectedSong.id).gte("line_index", payload.length),
    "多余歌词删除",
  );
  if (deleteTailResponse.error) {
    throw new Error(`歌词保存部分成功，但多余旧行删除失败：${deleteTailResponse.error.message}`);
  }
  const deleteNegativeResponse = await withSubmitTimeout(
    client.from("lyric_lines").delete().eq("song_id", state.selectedSong.id).lt("line_index", 0),
    "旧歌词删除",
  );
  if (deleteNegativeResponse.error) {
    throw new Error(`歌词保存部分成功，但旧歌词清理失败：${deleteNegativeResponse.error.message}`);
  }
  const refreshResponse = await withSubmitTimeout(
    client.from("lyric_lines").select("*").eq("song_id", state.selectedSong.id).order("line_index", { ascending: true }),
    "歌词刷新",
  );
  if (refreshResponse.error) throw refreshResponse.error;
  state.lyrics = refreshResponse.data || [];
  if (!lyricRowsMatchPayload(state.lyrics, payload)) {
    throw new Error("歌词保存后数据库校验不一致，请刷新页面核对后再继续编辑");
  }
}

function lyricRowsMatchPayload(rows, payload) {
  if (rows.length !== payload.length) return false;
  return payload.every((expected, index) => {
    const actual = rows[index];
    return (
      actual &&
      String(actual.song_id) === String(expected.song_id) &&
      Number(actual.line_index) === expected.line_index &&
      Math.abs(Number(actual.start_time_sec) - expected.start_time_sec) < TIME_EPSILON &&
      Math.abs(Number(actual.end_time_sec) - expected.end_time_sec) < TIME_EPSILON &&
      String(actual.text || "") === String(expected.text || "")
    );
  });
}

function updateActiveLyric(options = {}) {
  if (!els.lyricsViewport) return;
  const shouldScroll = options.scroll === true || (!els.lyricsModal.classList.contains("open") && options.scroll !== false);
  const current = Number(els.audioPlayer.currentTime || 0);
  const lines = lyricRowsForCurrentPlayback();
  if (!lines.length) return;
  const index = activeLyricIndexForTime(lines, current);
  if (index < 0) return;
  if (index === state.currentLyricIndex && !options.force) return;
  state.currentLyricIndex = index;
  updateCompactLyricActiveState(index, shouldScroll);
  updateLyricModalActive({ scroll: false });
}

function updateCompactLyricActiveState(index, shouldScroll) {
  for (const node of els.lyricsViewport.querySelectorAll(".lyric-scroll-line")) {
    node.classList.toggle("active", Number(node.dataset.lyricIndex) === index);
  }
  const active = els.lyricsViewport.querySelector(".lyric-scroll-line.active");
  if (active && shouldScroll) {
    const top = active.offsetTop - els.lyricsViewport.clientHeight / 2 + active.clientHeight / 2;
    els.lyricsViewport.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
  }
}

function lyricRowsForCurrentPlayback() {
  if (els.lyricsModal?.classList.contains("open") && els.lyricsBulkList) {
    return readLyricBulkRowsForPlayback();
  }
  return state.lyrics;
}

function readLyricBulkRowsForPlayback() {
  return [...(els.lyricsBulkList?.querySelectorAll(".lyric-bulk-row") || [])].map((row) => {
    const startText = row.querySelector(".lyric-bulk-start")?.value;
    const endText = row.querySelector(".lyric-bulk-end")?.value;
    const text = row.querySelector(".lyric-bulk-text")?.value.trim() || "";
    return {
      start_time_sec: parseOptionalLyricTime(startText),
      end_time_sec: parseOptionalLyricTime(endText),
      text,
    };
  });
}

function parseOptionalLyricTime(value) {
  try {
    return parseTimeInput(value);
  } catch {
    return NaN;
  }
}

function activeLyricIndexForTime(lines, current) {
  let activeIndex = -1;
  let firstValidIndex = -1;
  for (let index = 0; index < lines.length; index += 1) {
    const start = Number(lines[index].start_time_sec);
    if (!Number.isFinite(start)) continue;
    if (firstValidIndex < 0) firstValidIndex = index;
    if (current + TIME_EPSILON >= start) {
      activeIndex = index;
      continue;
    }
    if (activeIndex >= 0) break;
  }
  return activeIndex >= 0 ? activeIndex : firstValidIndex;
}

function setZoom(value) {
  state.zoom = Math.max(1, Math.min(8, Number(value || 1)));
  els.zoomRange.value = state.zoom;
  renderTimeline();
}

function startTimelineDrag(event) {
  if (event.button !== 0) return;
  state.timelineDrag = {
    startX: event.clientX,
    scrollLeft: els.timeline.scrollLeft,
    moved: false,
  };
  els.timeline.classList.add("dragging");
  els.timeline.setPointerCapture?.(event.pointerId);
}

function moveTimelineDrag(event) {
  if (!state.timelineDrag) return;
  const delta = event.clientX - state.timelineDrag.startX;
  if (Math.abs(delta) > 4) state.timelineDrag.moved = true;
  els.timeline.scrollLeft = state.timelineDrag.scrollLeft - delta;
}

function endTimelineDrag(event) {
  if (!state.timelineDrag) return;
  els.timeline.releasePointerCapture?.(event.pointerId);
  els.timeline.classList.remove("dragging");
  setTimeout(() => {
    state.timelineDrag = null;
  }, 0);
}

function riskLabel(section) {
  if (section.need_human_review) return "review";
  if (Number(section.type_confidence) < 0.74) return "type";
  if (Number(section.boundary_confidence) < 0.7) return "edge";
  return "ok";
}

function qcStatusText(status) {
  return status === "human_modified" ? "已修改" : "认同机器";
}

function sectionTypeTitle(type) {
  return String(type || "section").replace("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function resolveAudioPath(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  if (window.location.protocol === "file:") {
    return path.startsWith("/songs/") ? `..${path}` : path;
  }
  return path;
}

const traditionalToSimplifiedMap = {
  體: "体", 認: "认", 標: "标", 記: "记", 結: "结", 構: "构", 對: "对", 應: "应", 開: "开", 關: "关",
  顯: "显", 示: "示", 選: "选", 擇: "择", 寫: "写", 當: "当", 前: "前", 見: "见", 過: "过", 卻: "却",
  終: "终", 夢: "梦", 裡: "里", 難: "难", 頭: "头", 個: "个", 還: "还", 裝: "装", 雲: "云", 著: "着",
  動: "动", 承: "承", 煩: "烦", 輕: "轻", 愛: "爱", 與: "与", 紅: "红", 這: "这", 腳: "脚", 灑: "洒",
  脫: "脱", 說: "说", 聲: "声", 麼: "么", 長: "长", 許: "许", 讓: "让", 雙: "双", 遠: "远", 追: "追",
  來: "来", 會: "会", 從: "从", 為: "为", 無: "无", 後: "后", 風: "风", 點: "点", 聽: "听", 時: "时",
  間: "间", 洪: "洪", 流: "流", 程: "程", 響: "响", 歡: "欢", 歲: "岁", 靈: "灵", 寫: "写", 萬: "万",
  閃: "闪", 發: "发", 劃: "划", 舊: "旧", 新: "新", 負: "负", 敗: "败", 勝: "胜", 懂: "懂", 變: "变",
  願: "愿", 離: "离", 亂: "乱", 經: "经", 線: "线", 樣: "样", 像: "像", 將: "将", 請: "请", 點: "点",
  號: "号", 頁: "页", 場: "场", 語: "语", 詞: "词", 斷: "断", 區: "区", 實: "实", 質: "质", 檢: "检",
  們: "们", 兌: "兑", 別: "别", 剋: "克", 參: "参", 嗎: "吗", 堂: "堂", 尋: "寻", 張: "张", 憂: "忧",
  憶: "忆", 懷: "怀", 擔: "担", 數: "数", 會: "会", 歲: "岁", 樣: "样", 淚: "泪", 溫: "温", 滿: "满",
  濕: "湿", 爭: "争", 現: "现", 畫: "画", 盡: "尽", 眸: "眸", 給: "给", 綱: "纲", 腦: "脑", 臉: "脸",
  處: "处", 虛: "虚", 親: "亲", 諾: "诺", 謝: "谢", 護: "护", 讀: "读", 詞: "词", 遙: "遥", 遲: "迟",
  邊: "边", 錯: "错", 閉: "闭", 闊: "阔", 陽: "阳", 雪: "雪", 露: "露", 高: "高", 鬧: "闹", 藍: "蓝",
};

function toSimplified(text) {
  return String(text || "")
    .split("")
    .map((char) => traditionalToSimplifiedMap[char] || char)
    .join("");
}

function typeColor(type) {
  return getComputedStyle(document.documentElement).getPropertyValue(`--${type}`) || "#17202a";
}

function displayRange(row) {
  return `${displayTime(row.start_time_sec, row.start_time_label)}-${displayTime(row.end_time_sec, row.end_time_label)}`;
}

function displayTime(value) {
  return formatTime(value);
}

function timeToSeconds(value, fallback = 0) {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "number") return Number.isFinite(value) ? value : fallback;
  const text = String(value).trim();
  if (!text) return fallback;
  const direct = Number(text);
  if (Number.isFinite(direct)) return direct;
  if (text.includes(":")) {
    const parts = text.split(":").map((part) => Number(part));
    if (parts.length === 2 && parts.every(Number.isFinite)) {
      return parts[0] * 60 + parts[1];
    }
    if (parts.length === 3 && parts.every(Number.isFinite)) {
      return parts[0] * 60 + parts[1] + parts[2] / 100;
    }
  }
  return fallback;
}

function formatTime(value) {
  const totalCentiseconds = Math.max(0, Math.round(timeToSeconds(value, 0) * 100));
  const minutes = Math.floor(totalCentiseconds / 6000);
  const seconds = String(Math.floor((totalCentiseconds % 6000) / 100)).padStart(2, "0");
  const centiseconds = String(totalCentiseconds % 100).padStart(2, "0");
  return `${minutes}:${seconds}:${centiseconds}`;
}

function parseTimeInput(value) {
  const text = String(value || "").trim();
  if (!text) throw new Error("时间不能为空");
  if (!text.includes(":")) {
    const seconds = Number(text);
    if (!Number.isFinite(seconds) || seconds < 0) throw new Error(`无法识别时间：${text}`);
    return seconds;
  }
  const parts = text.split(":");
  if (parts.length !== 2 && parts.length !== 3) throw new Error(`时间格式应为 M:SS:CC，例如 3:12:56`);
  const minutes = Number(parts[0]);
  const seconds = Number(parts[1]);
  const centiseconds = parts.length === 3 ? Number(parts[2]) : 0;
  if (
    !Number.isFinite(minutes) ||
    !Number.isFinite(seconds) ||
    !Number.isFinite(centiseconds) ||
    minutes < 0 ||
    seconds < 0 ||
    centiseconds < 0 ||
    seconds >= 60 ||
    centiseconds >= 100
  ) {
    throw new Error(`时间格式应为 M:SS:CC，例如 3:12:56`);
  }
  return minutes * 60 + seconds + centiseconds / 100;
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

async function withSubmitTimeout(promise, action = "提交") {
  let timeoutId;
  try {
    return await Promise.race([
      Promise.resolve(promise),
      new Promise((_, reject) => {
        timeoutId = setTimeout(() => {
          reject(new Error(`${action}超时，请检查网络后重试`));
        }, SUBMIT_TIMEOUT_MS);
      }),
    ]);
  } finally {
    clearTimeout(timeoutId);
  }
}

function submitErrorMessage(error, action = "提交失败") {
  return `${action}：${error?.message || String(error)}`;
}

function showToast(message, isError = false, options = {}) {
  if (state.toastTimer) {
    clearTimeout(state.toastTimer);
    state.toastTimer = null;
  }
  els.toast.textContent = message;
  els.toast.style.background = isError ? "#bb2b2b" : "#17202a";
  els.toast.classList.add("visible");
  if (options.persistent) return;
  state.toastTimer = setTimeout(() => {
    els.toast.classList.remove("visible");
    state.toastTimer = null;
  }, 2200);
}

function togglePlayback() {
  if (els.audioPlayer.paused) els.audioPlayer.play().catch(() => showToast("音频暂时无法播放", true));
  else els.audioPlayer.pause();
}

function syncBottomPlayerVisibility() {
  const rect = document.querySelector(".transport")?.getBoundingClientRect();
  const shouldShow = Boolean(rect && rect.bottom < 0 && state.selectedSong);
  els.bottomPlayer.classList.toggle("visible", shouldShow);
  els.bottomPlayer.setAttribute("aria-hidden", shouldShow ? "false" : "true");
}

function isTypingTarget(target) {
  return ["INPUT", "TEXTAREA", "SELECT"].includes(target?.tagName) || target?.isContentEditable;
}

els.refreshButton.addEventListener("click", refreshReviewData);
els.statusFilter.addEventListener("change", renderSongPicker);
els.songDirectoryTitle.addEventListener("click", () => {
  state.directoryOpen = !state.directoryOpen;
  renderSongPicker();
});
for (const card of document.querySelectorAll("[data-song-filter]")) {
  const applyCardFilter = () => setSongFilter(card.dataset.songFilter);
  card.addEventListener("click", applyCardFilter);
}
els.songPicker.addEventListener("input", () => {
  const exact = state.songOptionMap.get(els.songPicker.value);
  if (exact) {
    selectSong(exact);
    return;
  }
  renderSongPicker();
});
els.songPicker.addEventListener("change", () => {
  const exact = state.songOptionMap.get(els.songPicker.value);
  if (exact) selectSong(exact);
});
els.songPicker.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  const text = els.songPicker.value.toLowerCase();
  const exact = state.songOptionMap.get(els.songPicker.value);
  if (exact) return selectSong(exact);
  const match = state.songs.find((song) => `${song.title} ${song.artist || ""} ${song.id}`.toLowerCase().includes(text));
  if (match) selectSong(match.id);
});
els.reviewFilter.addEventListener("change", renderSections);
els.finishReview.addEventListener("click", finishReview);
els.prevSection?.addEventListener("click", () => selectRelativeSection(-1));
els.nextSection?.addEventListener("click", () => selectRelativeSection(1));
els.useModel?.addEventListener("click", restoreModelValues);
els.addSection.addEventListener("click", addSectionAfterCurrent);
els.deleteSection.addEventListener("click", deleteCurrentSection);
els.resetCurrentSection?.addEventListener("click", resetCurrentSection);
els.resetSong?.addEventListener("click", resetWholeSong);
els.markPendingReview?.addEventListener("click", markSongPendingReviewOnly);
els.seekStart.addEventListener("click", () => {
  if (!state.selectedSection) return;
  try {
    locatePlaybackAt(parseTimeInput(els.editStart.value), { play: true, scrollSection: true });
  } catch (error) {
    showToast(error.message, true);
  }
});
els.lyricsExpand.addEventListener("click", openLyricsModal);
els.lyricsModalPlay.addEventListener("click", togglePlayback);
els.lyricsModalClose.addEventListener("click", closeLyricsModal);
els.lyricsModal.querySelector(".lyrics-modal-head")?.addEventListener("pointerdown", startLyricModalMove);
els.lyricsModalResize?.addEventListener("pointerdown", startLyricModalResize);
els.lyricsBulkAdd.addEventListener("click", addLyricBulkRow);
els.lyricsBulkSave.addEventListener("click", saveBulkLyrics);
els.lyricsModal.addEventListener("click", (event) => {
  if (event.target === els.lyricsModal) closeLyricsModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && els.lyricsModal.classList.contains("open")) closeLyricsModal();
});
els.audioPlayer.addEventListener("timeupdate", updateTimelinePlayhead);
els.audioPlayer.addEventListener("loadedmetadata", () => {
  setPlaybackRate(state.playbackRate);
  updateTransport();
  renderTimeline();
  syncBottomPlayerVisibility();
});
els.audioPlayer.addEventListener("play", updateTransport);
els.audioPlayer.addEventListener("pause", updateTransport);
els.playToggle.addEventListener("click", togglePlayback);
els.bottomPlayToggle.addEventListener("click", togglePlayback);
els.speedToggle?.addEventListener("click", cyclePlaybackRate);
els.bottomSpeedToggle?.addEventListener("click", cyclePlaybackRate);
els.progressRange.addEventListener("pointerdown", () => {
  state.isScrubbing = true;
  state.previewEndTime = null;
});
els.progressRange.addEventListener("input", () => {
  locatePlaybackAt(Number(els.progressRange.value || 0), { scrollSection: true });
});
els.progressRange.addEventListener("change", () => {
  state.isScrubbing = false;
  updateTimelinePlayhead();
});
els.progressRange.addEventListener("pointerup", () => {
  state.isScrubbing = false;
});
els.bottomProgressRange.addEventListener("pointerdown", () => {
  state.isScrubbing = true;
  state.previewEndTime = null;
});
els.bottomProgressRange.addEventListener("input", () => {
  locatePlaybackAt(Number(els.bottomProgressRange.value || 0), { scrollSection: true });
});
els.bottomProgressRange.addEventListener("change", () => {
  state.isScrubbing = false;
  updateTimelinePlayhead();
});
els.bottomProgressRange.addEventListener("pointerup", () => {
  state.isScrubbing = false;
});
window.addEventListener("scroll", syncBottomPlayerVisibility, { passive: true });
window.addEventListener("resize", syncBottomPlayerVisibility);
document.addEventListener("keydown", (event) => {
  if (handleSectionSubmitShortcut(event)) return;
  if (event.code !== "Space") return;
  if (isTypingTarget(event.target)) return;
  if (!state.selectedSong && !els.lyricsModal.classList.contains("open")) return;
  event.preventDefault();
  togglePlayback();
});
els.timeline.addEventListener("click", (event) => {
  if (event.target.classList.contains("timeline-segment")) return;
  seekFromTimeline(event, sectionTotalDuration(), true);
});
els.timeline.addEventListener("pointerdown", startTimelineDrag);
els.timeline.addEventListener("pointermove", moveTimelineDrag);
els.timeline.addEventListener("pointerup", endTimelineDrag);
els.timeline.addEventListener("pointercancel", endTimelineDrag);
els.zoomRange.addEventListener("input", () => setZoom(els.zoomRange.value));
els.zoomOut.addEventListener("click", () => setZoom(state.zoom - 0.5));
els.zoomIn.addEventListener("click", () => setZoom(state.zoom + 0.5));
els.editType.addEventListener("change", renderTypeButtonState);
for (const input of [els.editType, els.editStart, els.editEnd]) {
  input.addEventListener("focus", () => rememberEditorState("编辑 section"));
  input.addEventListener("input", () => {
    renderTypeButtonState();
  });
}
document.addEventListener("keydown", (event) => {
  const isUndo = (event.metaKey || event.ctrlKey) && !event.shiftKey && event.key.toLowerCase() === "z";
  if (!isUndo) return;
  if (els.lyricsModal.classList.contains("open")) {
    event.preventDefault();
    undoLastLyricAction();
    return;
  }
  if (isTypingTarget(event.target) && !els.editForm.contains(event.target)) return;
  event.preventDefault();
  undoLastEditorAction();
});
function handleSectionSubmitShortcut(event) {
  if (els.lyricsModal.classList.contains("open")) return false;
  if (!state.selectedSection) return false;
  const isSpace = event.code === "Space";
  const isEnter = event.key === "Enter";
  if (!isSpace && !isEnter) return false;
  const typeButton = event.target.closest?.("[data-type]");
  if (!els.editForm.contains(event.target) && !els.typeQuickButtons.contains(event.target)) return false;
  if (event.target === els.editStart || event.target === els.editEnd) return false;
  if (isEnter && !els.typeQuickButtons.contains(event.target) && event.target !== els.editType) return false;
  event.preventDefault();
  if (typeButton && els.typeQuickButtons.contains(typeButton)) {
    rememberEditorState("修改 section type");
    els.editType.value = typeButton.dataset.type;
    renderTypeButtonState();
  }
  els.editForm.requestSubmit();
  return true;
}

els.editForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const section = state.selectedSection;
  if (!section) return;
  let start;
  let end;
  try {
    start = parseTimeInput(els.editStart.value);
    end = parseTimeInput(els.editEnd.value);
  } catch (error) {
    return showToast(error.message, true);
  }
  const unchanged =
    !section.isLocal &&
    els.editType.value === section.section_type &&
    Math.abs(start - Number(section.start_time_sec)) < 0.5 &&
    Math.abs(end - Number(section.end_time_sec)) < 0.5;
  saveAnnotation(unchanged ? "approve" : "edit");
});

renderTypeButtons();
loadSongs().catch((error) => showToast(error.message, true));
