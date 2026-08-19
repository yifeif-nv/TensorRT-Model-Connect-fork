/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

"use strict";

const state = {
  token: "", cases: [], activeCase: null, chronology: [], inspectedPageRecords: new Set(),
};
const byId = (id) => document.getElementById(id);

document.addEventListener("DOMContentLoaded", () => {
  state.token = sessionStorage.getItem("evidence-token") || "";
  byId("reviewer-name").value = sessionStorage.getItem("evidence-reviewer") || "";
  bindEvents();
  if (state.token) connect(state.token);
});

function bindEvents() {
  byId("auth-form").addEventListener("submit", (event) => {
    event.preventDefault();
    connect(byId("token").value.trim());
  });
  byId("refresh-cases").addEventListener("click", loadCases);
  byId("create-case-form").addEventListener("submit", createCase);
  byId("upload-form").addEventListener("submit", uploadDocuments);
  byId("search-form").addEventListener("submit", runSearch);
  byId("build-chronology").addEventListener("click", buildChronology);
  byId("verify-case").addEventListener("click", verifyCase);
  byId("export-draft").addEventListener("click", () => exportCase(true));
  byId("export-case").addEventListener("click", () => exportCase(false));
  byId("reviewer-name").addEventListener("change", () => {
    sessionStorage.setItem("evidence-reviewer", byId("reviewer-name").value.trim());
  });
}

async function connect(token) {
  if (!token) return toast("Enter the local bearer token.");
  state.token = token;
  try {
    await api("/api/health");
    sessionStorage.setItem("evidence-token", token);
    byId("auth-panel").hidden = true;
    byId("workspace").hidden = false;
    byId("connection-dot").classList.add("connected");
    byId("connection-text").textContent = "Local workspace connected";
    await loadCases();
  } catch (error) {
    sessionStorage.removeItem("evidence-token");
    byId("connection-text").textContent = "Connection failed";
    toast(error.message);
  }
}

async function loadCases() {
  const payload = await api("/api/cases");
  state.cases = payload.cases;
  const list = byId("case-list");
  list.replaceChildren();
  for (const item of state.cases) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `case-button${state.activeCase?.id === item.id ? " active" : ""}`;
    const name = document.createElement("span");
    name.textContent = item.name;
    const detail = document.createElement("small");
    detail.textContent = item.head_snapshot_id ? `${item.coverage?.pages_total || 0} pages` : "empty case";
    button.append(name, detail);
    button.addEventListener("click", () => selectCase(item.id));
    list.append(button);
  }
}

async function createCase(event) {
  event.preventDefault();
  const name = byId("case-name").value.trim();
  const id = byId("case-id").value.trim();
  const created = await api("/api/cases", {
    method: "POST",
    body: JSON.stringify(id ? { name, id } : { name }),
  });
  event.target.reset();
  await loadCases();
  await selectCase(created.id);
}

async function selectCase(caseId) {
  state.activeCase = await api(`/api/cases/${encodeURIComponent(caseId)}`);
  byId("empty-state").hidden = true;
  byId("case-view").hidden = false;
  renderCase();
  await loadCases();
}

function renderCase() {
  const current = state.activeCase;
  byId("active-case-name").textContent = current.name;
  byId("snapshot-id").textContent = current.head_snapshot_id
    ? `snapshot ${current.head_snapshot_id}`
    : "No snapshot yet";
  const coverage = current.effective_coverage || current.manifest?.coverage || {};
  byId("metric-documents").textContent = coverage.documents_total || 0;
  byId("metric-pages").textContent = coverage.pages_total || 0;
  byId("metric-readable").textContent = coverage.pages_readable || 0;
  byId("metric-review").textContent = coverage.review_pending_pages?.length ?? coverage.pages_needing_review ?? 0;
  byId("metric-failed").textContent = coverage.pages_failed || 0;
  byId("metric-doc-failed").textContent = coverage.documents_failed || 0;
  byId("metric-excluded").textContent = coverage.documents_excluded || 0;
  byId("coverage-warning").hidden = !current.head_snapshot_id || Boolean(coverage.complete_for_negative_assertions);
  const sourceList = byId("source-list");
  sourceList.replaceChildren();
  for (const source of current.manifest?.sources || []) {
    const row = document.createElement("tr");
    const sourceCell = document.createElement("td");
    const open = document.createElement("button");
    open.type = "button";
    open.className = "secondary";
    open.textContent = evidenceAlias(source);
    open.addEventListener("click", () => openSource({ ...source, page_number: 1 }));
    const exclude = document.createElement("button");
    exclude.type = "button";
    exclude.className = "secondary";
    exclude.textContent = "Exclude";
    exclude.addEventListener("click", async () => {
      const reviewer = byId("reviewer-name").value.trim();
      if (!reviewer) return toast("Enter a reviewer name first.");
      const reason = window.prompt(`Why should ${evidenceAlias(source)} be excluded from the next snapshot?`);
      if (!reason) return;
      await api(`/api/cases/${encodeURIComponent(state.activeCase.id)}/documents/exclude`, {
        method: "POST",
        body: JSON.stringify({ document_id: source.document_id, reviewer, reason }),
      });
      toast("Document excluded in a new audited snapshot; prior snapshots remain available.");
      await selectCase(state.activeCase.id);
    });
    sourceCell.append(open, exclude);
    row.append(
      sourceCell,
      cell(source.source_sha256, true),
      cell(String(source.page_count)),
      cell(source.status),
    );
    sourceList.append(row);
  }
  for (const tombstone of current.manifest?.excluded_sources || []) {
    const source = tombstone.document;
    const row = document.createElement("tr");
    const sourceCell = document.createElement("td");
    const open = document.createElement("button");
    open.type = "button";
    open.className = "secondary";
    open.textContent = evidenceAlias(source);
    open.addEventListener("click", () => openSource({ ...source, page_number: 1 }));
    sourceCell.append(open);
    row.append(
      sourceCell,
      cell(source.source_sha256, true),
      cell(String(source.page_count)),
      cell(`EXCLUDED — ${tombstone.reason} (${tombstone.reviewer})`),
    );
    sourceList.append(row);
  }
  renderPageReviews();
}

function renderPageReviews() {
  const container = byId("page-review-list");
  container.replaceChildren();
  for (const source of state.activeCase?.manifest?.sources || []) {
    for (const page of source.pages || []) {
      if (!page.needs_review) continue;
      const targetId = `${source.document_id}:p${page.page_number}`;
      const review = state.activeCase.page_reviews?.[targetId] || {};
      container.append(reviewRow({
        label: `${evidenceAlias(source)} p.${page.page_number} · record ${page.record_sha256.slice(0, 12)}`,
        targetId,
        targetType: "page",
        recordHash: page.record_sha256,
        status: review.status || "unreviewed",
        notes: review.notes || "",
        onOpen: () => openPageReview(source, page),
      }));
    }
  }
}

async function uploadDocuments(event) {
  event.preventDefault();
  if (!state.activeCase) return;
  const files = [...byId("documents").files];
  const progress = byId("upload-progress");
  progress.replaceChildren();
  for (const file of files) {
    const item = document.createElement("div");
    item.className = "progress-item";
    const label = document.createElement("span");
    label.textContent = file.name;
    const status = document.createElement("span");
    status.textContent = "ingesting…";
    item.append(label, status);
    progress.append(item);
    try {
      const response = await fetch(`/api/cases/${encodeURIComponent(state.activeCase.id)}/documents`, {
        method: "POST",
        headers: { Authorization: `Bearer ${state.token}`, "X-Filename": file.name },
        body: file,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `Upload failed (${response.status})`);
      status.textContent = payload.document.status;
      if (payload.document.status !== "indexed") item.classList.add("error");
    } catch (error) {
      status.textContent = error.message;
      item.classList.add("error");
    }
  }
  byId("upload-form").reset();
  await selectCase(state.activeCase.id);
}

async function runSearch(event) {
  event.preventDefault();
  if (!state.activeCase?.head_snapshot_id) return toast("Ingest evidence before searching.");
  const query = byId("search-query").value.trim();
  const mode = byId("search-mode").value;
  const result = await api(`/api/cases/${encodeURIComponent(state.activeCase.id)}/search`, {
    method: "POST", body: JSON.stringify({ query, mode, limit: 20 }),
  });
  const status = byId("search-status");
  status.textContent = `${result.status} — ${result.matches.length} result(s). ${result.boundary}`;
  const container = byId("search-results");
  container.replaceChildren();
  for (const match of result.matches) {
    const card = document.createElement("article");
    card.className = "result-card";
    const header = document.createElement("header");
    const title = document.createElement("strong");
    title.textContent = match.label;
    const actions = document.createElement("div");
    const open = document.createElement("button");
    open.type = "button";
    open.className = "secondary";
    open.textContent = "Open source";
    open.addEventListener("click", () => openSource(match));
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "secondary";
    copy.textContent = "Copy citation";
    copy.addEventListener("click", () => navigator.clipboard.writeText(`${match.label} ${match.quote}`));
    actions.append(open, copy);
    header.append(title, actions);
    const quote = document.createElement("blockquote");
    quote.textContent = match.quote;
    card.append(header, quote);
    if (match.needs_review) {
      const review = document.createElement("p");
      review.className = "review";
      review.textContent = "OCR or low-quality extraction — review the source page.";
      card.append(review);
    }
    container.append(card);
  }
}

async function openSource(match) {
  const evidencePath = match.evidence_image
    ? `/api/cases/${encodeURIComponent(state.activeCase.id)}/page-image/${match.evidence_image.split("/").pop()}`
    : `/api/cases/${encodeURIComponent(state.activeCase.id)}/source/${match.document_id}`;
  const response = await fetch(
    evidencePath,
    { headers: { Authorization: `Bearer ${state.token}` } },
  );
  if (!response.ok) return toast("Could not open the source file.");
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  if (!match.evidence_image && [".html", ".htm", ".csv", ".json"].some((suffix) => match.filename.toLowerCase().endsWith(suffix))) {
    const download = document.createElement("a");
    download.href = url;
    download.download = match.filename;
    download.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 10000);
    return;
  }
  const opened = window.open(match.filename.toLowerCase().endsWith(".pdf") ? `${url}#page=${match.page_number}` : url, "_blank", "noopener");
  if (!opened) toast("The browser blocked the source window.");
  window.setTimeout(() => URL.revokeObjectURL(url), 120000);
}

async function openPageReview(source, page) {
  const route = `/api/cases/${encodeURIComponent(state.activeCase.id)}/page-record/${source.document_id}/p${page.page_number}`;
  const payload = await api(route);
  const record = payload.record;
  if (payload.record_sha256 !== page.record_sha256) {
    return toast("The page record changed; refresh the case before reviewing.");
  }
  if (payload.snapshot_id !== state.activeCase.head_snapshot_id) {
    return toast("The evidence snapshot changed; refresh the case before reviewing.");
  }

  state.inspectedPageRecords.add(`${payload.citation_id}:${payload.record_sha256}`);
  byId("review-dialog-title").textContent = `${evidenceAlias(source)} p.${page.page_number}`;
  byId("review-record-id").textContent = `Approving this exact record: SHA-256 ${payload.record_sha256}`;
  byId("review-page-text").textContent = record.text || "(no indexed text)";
  const metadata = byId("review-page-metadata");
  metadata.replaceChildren();
  for (const [term, value] of [
    ["Extraction", record.extraction_method],
    ["Status", record.status],
    ["Deterministic quality score", `${record.quality_score} (${record.quality_score_kind})`],
    ["Text SHA-256", record.text_sha256],
    ["Quality signals", JSON.stringify(record.metadata?.quality_signals || record.metadata?.native_quality_signals || {})],
  ]) {
    const key = document.createElement("dt");
    key.textContent = term;
    const detail = document.createElement("dd");
    detail.textContent = String(value ?? "");
    metadata.append(key, detail);
  }

  const image = byId("review-page-image");
  const noImage = byId("review-no-image");
  image.hidden = true;
  image.removeAttribute("src");
  noImage.hidden = Boolean(record.evidence_image);
  if (record.evidence_image) {
    const response = await fetch(
      `/api/cases/${encodeURIComponent(state.activeCase.id)}/page-image/${record.evidence_image.split("/").pop()}`,
      { headers: { Authorization: `Bearer ${state.token}` } },
    );
    if (!response.ok) return toast("Could not load the authenticated page image.");
    image.src = URL.createObjectURL(await response.blob());
    image.onload = () => URL.revokeObjectURL(image.src);
    image.hidden = false;
  }
  byId("review-open-original").onclick = () => openSource({
    ...source, page_number: page.page_number, evidence_image: "",
  });
  byId("review-dialog").showModal();
}

async function buildChronology() {
  if (!state.activeCase?.head_snapshot_id) return toast("Ingest evidence first.");
  const result = await api(`/api/cases/${encodeURIComponent(state.activeCase.id)}/chronology`, {
    method: "POST", body: "{}",
  });
  const body = byId("chronology-results");
  body.replaceChildren();
  state.chronology = result.events;
  for (const event of result.events) {
    const row = document.createElement("tr");
    const citation = document.createElement("td");
    const open = document.createElement("button");
    open.type = "button";
    open.className = "secondary";
    open.textContent = event.label;
    open.addEventListener("click", () => openSource(event));
    citation.append(open);
    const review = document.createElement("td");
    review.append(reviewRow({
      label: "Review event",
      targetId: event.event_id,
      targetType: "event",
      status: event.review_status,
      notes: event.review_notes,
      compact: true,
    }));
    row.append(
      cell(event.normalized_date || event.raw_date),
      citation,
      cell(event.quote),
      review,
    );
    body.append(row);
  }
  toast(`Chronology contains ${result.events.length} cited date event(s).`);
}

async function verifyCase() {
  if (!state.activeCase?.head_snapshot_id) return toast("No snapshot to verify.");
  const result = await api(`/api/cases/${encodeURIComponent(state.activeCase.id)}/verify`);
  toast(result.ok ? `Verified ${result.snapshot_id}` : result.failures.join("; "));
}

async function exportCase(draft) {
  if (!state.activeCase?.head_snapshot_id) return toast("No snapshot to export.");
  try {
    const result = await api(`/api/cases/${encodeURIComponent(state.activeCase.id)}/export`, {
      method: "POST", body: JSON.stringify({ include_originals: true, draft }),
    });
    const response = await fetch(result.download_url, {
      headers: { Authorization: `Bearer ${state.token}` },
    });
    if (!response.ok) throw new Error("Export was created but download failed.");
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${state.activeCase.id}-${result.export_status}-evidence.zip`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 10000);
    toast(`${result.export_status.toUpperCase()} audit bundle downloaded.`);
  } catch (error) {
    toast(error.message);
  }
}

function reviewRow({
  label, targetId, targetType, status, notes, compact = false, onOpen = null, recordHash = "",
}) {
  const row = document.createElement("div");
  row.className = "review-row";
  const title = document.createElement(onOpen ? "button" : "strong");
  title.textContent = label;
  if (onOpen) {
    title.type = "button";
    title.className = "secondary";
    title.addEventListener("click", onOpen);
  }
  const select = document.createElement("select");
  for (const value of ["unreviewed", "accepted", "rejected"]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === status;
    select.append(option);
  }
  const note = document.createElement("input");
  note.placeholder = "Review notes";
  note.value = notes || "";
  const save = document.createElement("button");
  save.type = "button";
  save.textContent = "Save review";
  save.addEventListener("click", async () => {
    const reviewer = byId("reviewer-name").value.trim();
    if (!reviewer) return toast("Enter a reviewer name first.");
    if (
      targetType === "page"
      && select.value !== "unreviewed"
      && !state.inspectedPageRecords.has(`${targetId}:${recordHash}`)
    ) {
      return toast("Open the page comparison before recording a decision for this exact record.");
    }
    try {
      await api(`/api/cases/${encodeURIComponent(state.activeCase.id)}/reviews/${targetType}`, {
        method: "POST",
        body: JSON.stringify({
          snapshot_id: state.activeCase.head_snapshot_id,
          target_id: targetId,
          record_sha256: recordHash,
          status: select.value,
          reviewer,
          notes: note.value,
        }),
      });
    } catch (error) {
      toast(`${error.message} The case has been refreshed; inspect the page again.`);
      if (targetType === "page") await selectCase(state.activeCase.id);
      return;
    }
    toast("Review decision recorded in the audit chain.");
    if (targetType === "page") await selectCase(state.activeCase.id);
    else await buildChronology();
  });
  row.append(title, select, note, save);
  if (compact) row.classList.add("compact");
  return row;
}

function evidenceAlias(source) {
  return `${source.filename} · ${source.document_id.slice(0, 8)}`;
}

function cell(value, mono = false) {
  const td = document.createElement("td");
  if (mono) {
    const code = document.createElement("code");
    code.textContent = value;
    td.append(code);
  } else td.textContent = value;
  return td;
}

async function api(path, options = {}) {
  const headers = { Authorization: `Bearer ${state.token}`, ...(options.headers || {}) };
  if (options.body && typeof options.body === "string") headers["Content-Type"] = "application/json";
  const response = await fetch(path, { ...options, headers });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function toast(message) {
  const element = byId("toast");
  element.textContent = message;
  element.hidden = false;
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => { element.hidden = true; }, 7000);
}
