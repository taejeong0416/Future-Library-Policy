const el = (id) => document.getElementById(id);

const fileInput = el("fileInput");
const dropzone = el("dropzone");
const errorBox = el("errorBox");
const content = el("content");

const inputSummary = el("inputSummary");
const policyNotes = el("policyNotes");

const expr = el("expr");
const sem = el("sem");
const structBox = el("struct");
const info = el("info");
const infoDelta = el("infoDelta");

const exprRaw = el("exprRaw");
const semRaw = el("semRaw");
const structRaw = el("structRaw");
const infoRaw = el("infoRaw");
const rawJson = el("rawJson");

const btnExample = el("btnExample");
const btnClear = el("btnClear");

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function showContent() {
  content.classList.remove("hidden");
}

function hideContent() {
  content.classList.add("hidden");
}

function kvRow(key, value, badge) {
  const wrap = document.createElement("div");
  wrap.className = "row";
  const k = document.createElement("div");
  k.className = "k";
  k.textContent = key;

  const v = document.createElement("div");
  v.className = "v";

  if (badge) {
    v.innerHTML = badgeHTML(badge.level, badge.text);
  } else {
    v.textContent = String(value);
  }

  wrap.appendChild(k);
  wrap.appendChild(v);
  return wrap;
}

function badgeHTML(level, text) {
  const cls = level === "good" ? "good" : level === "warn" ? "warn" : "bad";
  return `<span class="badge"><span class="dot ${cls}"></span>${escapeHtml(text)}</span>`;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatNum(n, digits = 4) {
  if (typeof n !== "number" || Number.isNaN(n)) return "-";
  return n.toFixed(digits);
}

function formatInt(n) {
  if (typeof n !== "number" || Number.isNaN(n)) return "-";
  return String(n);
}

function scoreLevel(x) {
  // 단순 시각 표시용(판정 아님): 높/중/낮
  if (typeof x !== "number") return { level: "warn", text: "-" };
  if (x >= 0.7) return { level: "good", text: formatNum(x, 4) };
  if (x >= 0.4) return { level: "warn", text: formatNum(x, 4) };
  return { level: "bad", text: formatNum(x, 4) };
}

function percent(x) {
  if (typeof x !== "number" || Number.isNaN(x)) return "-";
  return (x * 100).toFixed(2) + "%";
}

function validateReport(obj) {
  // 최소 요구 스키마 체크
  if (!obj || typeof obj !== "object") throw new Error("JSON이 객체가 아닙니다.");
  if (!obj.metrics || typeof obj.metrics !== "object") throw new Error("metrics가 없습니다.");
  const m = obj.metrics;
  const need = ["expression_similarity", "semantic_similarity", "structural_change", "information_expansion_reduction"];
  for (const k of need) {
    if (!m[k]) throw new Error(`metrics.${k}가 없습니다.`);
  }
  return true;
}

function renderBins(bins) {
  const box = el("semBins");
  box.innerHTML = "";
  if (!bins || typeof bins !== "object") return;

  const entries = [
    [">=0.8", bins[">=0.8"] ?? 0],
    ["0.5~0.8", bins["0.5~0.8"] ?? 0],
    ["0.2~0.5", bins["0.2~0.5"] ?? 0],
    ["<0.2", bins["<0.2"] ?? 0],
  ];

  const total = entries.reduce((acc, [, c]) => acc + (Number(c) || 0), 0) || 1;

  for (const [label, count] of entries) {
    const row = document.createElement("div");
    row.className = "bin";

    const lab = document.createElement("div");
    lab.className = "label";
    lab.textContent = label;

    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.width = ((Number(count) || 0) / total) * 100 + "%";
    bar.appendChild(fill);

    const cnt = document.createElement("div");
    cnt.className = "count";
    cnt.textContent = String(count ?? 0);

    row.appendChild(lab);
    row.appendChild(bar);
    row.appendChild(cnt);
    box.appendChild(row);
  }
}

function render(report) {
  clearError();
  validateReport(report);

  const m = report.metrics;

  // 입력 요약
  inputSummary.innerHTML = "";
  const draftChars = report?.inputs?.draft_chars ?? m?.expression_similarity?.length?.chars_draft ?? "-";
  const finalChars = report?.inputs?.final_chars ?? m?.expression_similarity?.length?.chars_final ?? "-";
  const deltaChars = (typeof draftChars === "number" && typeof finalChars === "number") ? (finalChars - draftChars) : "-";

  inputSummary.appendChild(kvRow("draft_chars", draftChars));
  inputSummary.appendChild(kvRow("final_chars", finalChars));
  inputSummary.appendChild(kvRow("delta_chars", deltaChars));

  // 정책 메모
  policyNotes.innerHTML = "";
  const notes = Array.isArray(report.policy_notes) ? report.policy_notes : [];
  for (const n of notes) {
    const li = document.createElement("li");
    li.textContent = n;
    policyNotes.appendChild(li);
  }
  if (!notes.length) {
    const li = document.createElement("li");
    li.textContent = "policy_notes가 비어 있습니다.";
    policyNotes.appendChild(li);
  }

  // 표현 유사도
  expr.innerHTML = "";
  const exprM = m.expression_similarity;
  expr.appendChild(kvRow("char_similarity", "", scoreLevel(exprM.char_similarity)));
  expr.appendChild(kvRow("tokens(draft)", exprM?.token_counts?.draft ?? "-"));
  expr.appendChild(kvRow("tokens(final)", exprM?.token_counts?.final ?? "-"));
  expr.appendChild(kvRow("insert_rate(vs draft)", percent(exprM?.token_change_rates_vs_draft?.insert_rate)));
  expr.appendChild(kvRow("delete_rate(vs draft)", percent(exprM?.token_change_rates_vs_draft?.delete_rate)));
  expr.appendChild(kvRow("replace_rate(vs draft)", percent(exprM?.token_change_rates_vs_draft?.replace_rate)));
  expr.appendChild(kvRow("delta_chars", exprM?.length?.delta_chars ?? "-"));
  exprRaw.textContent = JSON.stringify(exprM, null, 2);

  // 의미 유사도
  sem.innerHTML = "";
  const semM = m.semantic_similarity;
  sem.appendChild(kvRow("doc_tfidf_cosine", "", scoreLevel(semM.doc_tfidf_cosine)));
  sem.appendChild(kvRow("para_count(draft)", semM?.para_counts?.draft ?? "-"));
  sem.appendChild(kvRow("para_count(final)", semM?.para_counts?.final ?? "-"));

  const dist = semM?.para_bestmatch_similarity_distribution;
  sem.appendChild(kvRow("para_sim_mean", dist?.mean ?? "-"));
  sem.appendChild(kvRow("para_sim_q10", dist?.q10 ?? "-"));
  sem.appendChild(kvRow("para_sim_q50", dist?.q50 ?? "-"));
  sem.appendChild(kvRow("para_sim_q90", dist?.q90 ?? "-"));

  renderBins(dist?.bins);
  semRaw.textContent = JSON.stringify(semM, null, 2);

  // 구조 변화도
  structBox.innerHTML = "";
  const stM = m.structural_change;
  structBox.appendChild(kvRow("heading_count(draft)", stM?.heading_counts?.draft ?? "-"));
  structBox.appendChild(kvRow("heading_count(final)", stM?.heading_counts?.final ?? "-"));
  structBox.appendChild(kvRow("common_heading_count", stM?.heading_set_changes?.common_count ?? "-"));
  structBox.appendChild(kvRow("order_stability_proxy", stM?.order_stability_proxy ?? "-"));
  if (stM?.note) structBox.appendChild(kvRow("note", stM.note));
  structRaw.textContent = JSON.stringify(stM, null, 2);

  // 정보 확장·축소
  info.innerHTML = "";
  infoDelta.innerHTML = "";
  const infoM = m.information_expansion_reduction;

  const d = infoM?.draft || {};
  const f = infoM?.final || {};
  const del = infoM?.delta || {};

  info.appendChild(kvRow("tokens(draft→final)", `${formatInt(d.tokens)} → ${formatInt(f.tokens)}`));
  info.appendChild(kvRow("unique_tokens(draft→final)", `${formatInt(d.unique_tokens)} → ${formatInt(f.unique_tokens)}`));
  info.appendChild(kvRow("numbers(draft→final)", `${formatInt(d.numbers)} → ${formatInt(f.numbers)}`));
  info.appendChild(kvRow("quotes(draft→final)", `${formatInt(d.quotes)} → ${formatInt(f.quotes)}`));
  info.appendChild(kvRow("paren_groups(draft→final)", `${formatInt(d.paren_groups)} → ${formatInt(f.paren_groups)}`));

  infoDelta.appendChild(kvRow("Δ tokens", del.tokens ?? "-"));
  infoDelta.appendChild(kvRow("Δ unique_tokens", del.unique_tokens ?? "-"));
  infoDelta.appendChild(kvRow("Δ numbers", del.numbers ?? "-"));
  infoDelta.appendChild(kvRow("Δ quotes", del.quotes ?? "-"));
  infoDelta.appendChild(kvRow("Δ paren_groups", del.paren_groups ?? "-"));

  if (infoM?.note) info.appendChild(kvRow("note", infoM.note));
  infoRaw.textContent = JSON.stringify(infoM, null, 2);

  // 원본 JSON
  rawJson.textContent = JSON.stringify(report, null, 2);

  showContent();
}

async function loadFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".json")) {
    showError("JSON 파일만 업로드해 주세요.");
    return;
  }
  try {
    const text = await file.text();
    const obj = JSON.parse(text);
    render(obj);
  } catch (e) {
    showError("JSON 파싱 또는 렌더링 실패:\n" + (e?.message || String(e)));
    hideContent();
  }
}

fileInput.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  loadFile(f);
  fileInput.value = "";
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const f = e.dataTransfer.files?.[0];
  loadFile(f);
});

// 예시 데이터: 네가 붙여준 JSON 형태를 그대로 축약 반영
btnExample.addEventListener("click", () => {
  const example = {
    inputs: { draft_chars: 1993, final_chars: 1890 },
    metrics: {
      expression_similarity: {
        char_similarity: 0.2833,
        token_counts: { draft: 482, final: 450 },
        token_change_rates_vs_draft: { insert_rate: 0.0145, delete_rate: 0.0145, replace_rate: 0.917 },
        length: { chars_draft: 1993, chars_final: 1890, delta_chars: -103 }
      },
      semantic_similarity: {
        doc_tfidf_cosine: 0.1841,
        para_counts: { draft: 1, final: 6 },
        para_bestmatch_similarity_distribution: {
          mean: 0.1263, q10: 0.1263, q50: 0.1263, q90: 0.1263,
          bins: { ">=0.8": 0, "0.5~0.8": 0, "0.2~0.5": 0, "<0.2": 1 }
        },
        top_para_matches: [{ draft_para: 0, final_para: 0, sim: 0.1263 }]
      },
      structural_change: {
        heading_counts: { draft: 0, final: 0 },
        heading_set_changes: { added_sample: [], removed_sample: [], common_count: 0 },
        order_stability_proxy: 0.0,
        note: "구조는 제목 패턴 기반의 단순 추정이며, 형식이 정돈될수록 정확도가 상승합니다."
      },
      information_expansion_reduction: {
        draft: { tokens: 482, unique_tokens: 382, numbers: 3, quotes: 8, paren_groups: 1, bracket_groups: 0, urls: 0, bullets: 0 },
        final: { tokens: 450, unique_tokens: 366, numbers: 3, quotes: 12, paren_groups: 0, bracket_groups: 0, urls: 0, bullets: 0 },
        delta: { tokens: -32, unique_tokens: -16, numbers: 0, quotes: 4, paren_groups: -1, bracket_groups: 0, urls: 0, bullets: 0 },
        ratio_final_over_draft: { tokens: 0.9336, unique_tokens: 0.9581, numbers: 1.0, quotes: 1.5, paren_groups: 0.0, bracket_groups: 0.0, urls: 0.0, bullets: 0.0 },
        note: "정보 확장/축소는 사실성·품질 판단이 아닌, 정보 요소의 증감 기록(프록시)입니다."
      }
    },
    policy_notes: [
      "본 리포트는 AI 사용 여부 판별, 기여도 평가, 품질 평가를 수행하지 않습니다.",
      "모든 지표는 초본 대비 최종본의 변형을 설명하는 기록 데이터입니다.",
      "총점·등급·우열 판단을 산출하지 않습니다."
    ]
  };
  render(example);
});

btnClear.addEventListener("click", () => {
  hideContent();
  clearError();
  rawJson.textContent = "";
});
