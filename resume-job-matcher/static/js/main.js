(function () {
  "use strict";

  const MAX_FILES = 10;
  const MAX_BYTES = 5 * 1024 * 1024;
  const ALLOWED = [".pdf", ".docx", ".txt"];
  const SAMPLE_RESUME_NAME = "sample_priya.txt";
  const SAMPLE_RESUME_TEXT = [
    "Priya Sharma",
    "priya.sharma@example.com | +91 98765 43210 | github.com/priyasharma",
    "Data Scientist with 5 years of experience building machine learning models.",
    "Skills: Python, SQL, pandas, NumPy, scikit-learn, XGBoost, Tableau, Apache Spark, statistics, A/B testing",
    "Experience",
    "- Built predictive models with scikit-learn and XGBoost and ran A/B testing experiments.",
    "- Created Tableau dashboards that tracked model performance for business teams.",
    "Education",
    "B.Tech in Computer Science, Pune University",
  ].join("\n");

  const $ = (id) => document.getElementById(id);
  const els = {
    status: $("status"), statusText: $("statusText"), statusDot: $("statusDot"),
    chipSpaCy: $("chipSpaCy"), chipEmbeddings: $("chipEmbeddings"), chipGemini: $("chipGemini"),
    jobSelect: $("jobSelect"), jobTitle: $("jobTitle"), jobText: $("jobText"),
    jobCounter: $("jobCounter"), jobLocation: $("jobLocation"),
    dropzone: $("dropzone"), fileInput: $("fileInput"), fileList: $("fileList"),
    matchingSampleBtn: $("matchingSampleBtn"), analyze: $("analyzeBtn"), clear: $("clearBtn"),
    error: $("errorBox"), results: $("results"), resultsEmpty: $("resultsEmpty"),
    resultsTitle: $("resultsTitle"), resultsMeta: $("resultsMeta"), warn: $("warnBox"),
    ranking: $("ranking"), jobSkills: $("jobSkills"), jobSkillCount: $("jobSkillCount"),
    weights: $("weights"), candidates: $("candidates"),
    jobLinksCard: $("jobLinksCard"), jobLinks: $("jobLinks"),
    jobLinksMeta: $("jobLinksMeta"), jobLinksDisclaimer: $("jobLinksDisclaimer"),
    aiCard: $("aiCard"), aiStatus: $("aiStatus"), aiTestBtn: $("aiTestBtn"),
    aiResumeSelect: $("aiResumeSelect"), aiConsent: $("aiConsent"),
    aiTaskButtons: $("aiTaskButtons"), aiError: $("aiError"),
    aiEmpty: $("aiEmpty"), aiResults: $("aiResults"),
    loading: $("loading"), loadingText: $("loadingText"),
  };

  let jobs = [];
  let files = [];
  let geminiInfo = null;
  let lastReport = null;
  let linksTimer = null;
  let aiBusy = false;

  // ---------------------------------------------------------------- helpers
  function esc(value) {
    const div = document.createElement("div");
    div.textContent = value == null ? "" : String(value);
    return div.innerHTML;
  }
  function escAttr(value) {
    return esc(value).replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function fmtSize(bytes) {
    if (!bytes) return "0 B";
    if (bytes < 1024) return bytes + " B";
    return bytes < 1024 * 1024 ? (bytes / 1024).toFixed(0) + " KB" : (bytes / 1024 / 1024).toFixed(1) + " MB";
  }
  function tone(score) { return score >= 60 ? "good" : score >= 40 ? "mid" : "low"; }
  function toneColor(score) { return score >= 60 ? "var(--good)" : score >= 40 ? "#e0a21b" : "var(--bad)"; }
  function pct(value) {
    return value == null || Number.isNaN(Number(value)) ? "n/a" : Number(value).toFixed(1) + "%";
  }
  function num(value, digits) {
    return value == null || Number.isNaN(Number(value)) ? "n/a" : Number(value).toFixed(digits == null ? 2 : digits);
  }
  function chips(items, cls, extra) {
    if (!items || !items.length) return '<span class="empty">None</span>';
    return items.map((i) => `<span class="chip ${cls || ""}">${esc(i.display || i)}${extra ? extra(i) : ""}</span>`).join("");
  }
  function joinOrEmpty(items, sep) {
    if (!items || !items.length) return '<span class="empty">not found</span>';
    return esc(items.join(sep || ", "));
  }
  function showAlert(el, message, details) {
    if (!el) return;
    let html = esc(message);
    if (details && details.length) {
      html += "<ul>" + details.map((d) => `<li><strong>${esc(d.filename || "Input")}</strong>: ${esc(d.error || d)}</li>`).join("") + "</ul>";
    }
    el.innerHTML = html;
    el.hidden = false;
  }
  function clearAlert(el) {
    if (!el) return;
    el.hidden = true;
    el.innerHTML = "";
  }
  function setEmpty(el, visible) { if (el) el.hidden = !visible; }
  function setButtonBusy(button, busy, busyText, idleText) {
    if (!button) return;
    if (busy) {
      button.dataset.idle = button.dataset.idle || button.textContent;
      button.disabled = true;
      button.innerHTML = `<span class="loading-inline" aria-hidden="true"></span>${esc(busyText || "Working…")}`;
    } else {
      button.disabled = false;
      button.textContent = idleText || button.dataset.idle || "Run";
    }
  }
  function setLoading(on, text) {
    els.loading.hidden = !on;
    if (text) els.loadingText.textContent = text;
  }
  async function apiFetch(url, options) {
    const res = await fetch(url, options);
    let data = {};
    try { data = await res.json(); } catch (_) { /* non-JSON response */ }
    if (!res.ok) {
      const error = new Error(data.error || `Request failed (${res.status}).`);
      error.status = res.status;
      error.data = data;
      throw error;
    }
    return data;
  }
  function metric(label, value, sub, cls) {
    return `<div class="metric ${cls || ""}"><span>${esc(label)}</span><b>${esc(value)}</b>${sub ? `<small>${esc(sub)}</small>` : ""}</div>`;
  }
  function updateCounter() {
    const words = els.jobText.value.trim().split(/\s+/).filter(Boolean).length;
    els.jobCounter.textContent = words + " words";
  }
  function effectiveJobText() {
    const title = els.jobTitle.value.trim();
    const description = els.jobText.value.trim();
    return title ? `${title}\n${description}` : description;
  }
  function currentLocation() {
    return els.jobLocation.value.trim() || "India";
  }

  // ---------------------------------------------------------- backend status
  function setModelChip(el, state, text) {
    if (!el) return;
    el.className = "model-chip is-" + state;
    el.innerHTML = `<span class="model-chip__dot"></span>${esc(text)}`;
  }
  async function loadStatus() {
    try {
      const data = await apiFetch("/api/health");
      geminiInfo = data.gemini || null;
      const nlp = data.nlp || {};
      const embedding = data.embedding || {
        state: data.embedding_model_loaded ? "loaded" : "idle",
        model: data.embedding_model,
      };
      const gem = data.gemini || {};
      const ner = !!nlp.ner_model;

      els.status.className = "status status--" + (ner ? "ok" : "warn");
      els.statusText.textContent = ner ? "Backend ready · local NLP available" : "Backend ready · spaCy NER limited";
      setModelChip(els.chipSpaCy, ner ? "ok" : nlp.available ? "warn" : "bad",
        ner ? "spaCy: NER ready" : nlp.available ? "spaCy: blank pipeline (no NER)" : "spaCy: unavailable");
      if (embedding.state === "loaded") {
        setModelChip(els.chipEmbeddings, "ok", `Embeddings: ${embedding.model} loaded`);
      } else if (embedding.state === "fallback") {
        setModelChip(els.chipEmbeddings, "warn", "Embeddings: TF-IDF fallback active");
      } else {
        setModelChip(els.chipEmbeddings, "idle", `Embeddings: ${embedding.model} loads on first analysis`);
      }
      setModelChip(els.chipGemini, gem.configured ? "ok" : "idle",
        gem.configured ? `Gemini: configured · ${gem.model}` : "Gemini: optional (not configured)");
      if (geminiInfo) renderAiStatus(geminiInfo);
    } catch (e) {
      els.status.className = "status status--bad";
      els.statusText.textContent = "Backend unreachable";
      setModelChip(els.chipSpaCy, "bad", "spaCy: status unavailable");
      setModelChip(els.chipEmbeddings, "bad", "Embeddings: status unavailable");
      setModelChip(els.chipGemini, "bad", "Gemini: status unavailable");
      els.aiStatus.className = "status-card bad";
      els.aiStatus.textContent = "Could not read backend status.";
    }
  }

  // ------------------------------------------------------------------- jobs
  async function loadJobs() {
    try {
      const data = await apiFetch("/api/jobs");
      jobs = data.jobs || [];
      els.jobSelect.innerHTML = '<option value="">— Custom role —</option>' +
        jobs.map((j) => `<option value="${escAttr(j.id)}">${esc(j.title)}</option>`).join("");
      if (jobs.length) {
        els.jobSelect.value = jobs[0].id;
        applyJob();
      }
    } catch (e) {
      els.jobSelect.innerHTML = '<option value="">Could not load sample roles</option>';
    }
  }
  function applyJob() {
    const job = jobs.find((j) => j.id === els.jobSelect.value);
    if (!job) return;
    els.jobTitle.value = job.title;
    els.jobText.value = job.description;
    updateCounter();
  }

  // ------------------------------------------------------------------ files
  function addFiles(list) {
    clearAlert(els.error);
    const problems = [];
    for (const f of Array.from(list)) {
      const ext = "." + f.name.split(".").pop().toLowerCase();
      if (!ALLOWED.includes(ext)) { problems.push({ filename: f.name, error: "Unsupported type. Use PDF, DOCX or TXT." }); continue; }
      if (!f.size) { problems.push({ filename: f.name, error: "File is empty." }); continue; }
      if (f.size > MAX_BYTES) { problems.push({ filename: f.name, error: "Larger than 5 MB." }); continue; }
      if (files.some((x) => x.name === f.name)) { problems.push({ filename: f.name, error: "Already added." }); continue; }
      if (files.length >= MAX_FILES) { problems.push({ filename: f.name, error: `Maximum ${MAX_FILES} files.` }); continue; }
      files.push(f);
    }
    renderFiles();
    if (problems.length) showAlert(els.error, "Some files were not added:", problems);
  }
  function renderFiles() {
    els.fileList.innerHTML = files.map((f, i) =>
      `<li><span class="fname">${esc(f.name)}<span class="fsize">${fmtSize(f.size)}</span></span>` +
      `<button type="button" data-remove="${i}" aria-label="Remove ${escAttr(f.name)}">×</button></li>`).join("");
    updateAiResumeOptions();
  }
  function loadSampleResume() {
    files = files.filter((f) => f.name !== SAMPLE_RESUME_NAME);
    addFiles([new File([SAMPLE_RESUME_TEXT], SAMPLE_RESUME_NAME, { type: "text/plain" })]);
  }
  function updateAiResumeOptions(preferred) {
    if (!els.aiResumeSelect) return;
    const top = lastReport && lastReport.candidates && lastReport.candidates[0]
      ? lastReport.candidates[0].filename : "";
    const current = preferred === undefined ? (els.aiResumeSelect.value || top) : preferred;
    els.aiResumeSelect.innerHTML = files.map((f) =>
      `<option value="${escAttr(f.name)}">${esc(f.name)}</option>`).join("");
    if (current && files.some((f) => f.name === current)) els.aiResumeSelect.value = current;
  }

  // ---------------------------------------------------------- matching flow
  async function analyze() {
    clearAlert(els.error);
    const title = els.jobTitle.value.trim();
    const description = els.jobText.value.trim();
    if (title.length < 2) return showAlert(els.error, "Enter a job title (for example, AI Engineer).");
    if (description.split(/\s+/).filter(Boolean).length < 10) return showAlert(els.error, "Enter a job description with at least 10 words.");
    if (!files.length) return showAlert(els.error, "Add at least one resume (PDF, DOCX or TXT).");

    const form = new FormData();
    form.append("job_title", title);
    form.append("job_description", effectiveJobText());
    files.forEach((f) => form.append("resumes", f, f.name));

    setLoading(true, "Extracting text, running NER, embeddings and TF-IDF…");
    try {
      const data = await apiFetch("/api/match", { method: "POST", body: form });
      renderResults(data);
      loadStatus();
    } catch (e) {
      showAlert(els.error, e.message || "Could not reach the server. Is the Flask app still running?", e.data && e.data.file_errors);
    } finally {
      setLoading(false);
    }
  }

  function entityRows(entities) {
    const exp = entities.experience || {};
    const rows = [
      ["Name", entities.name],
      ["Other persons", (entities.persons || []).filter((p) => p !== entities.name).join(", ")],
      ["Email", (entities.emails || []).join(", ")],
      ["Phone", (entities.phones || []).join(", ")],
      ["Relevant links", (entities.links || []).join(", ")],
      ["Education", (entities.education || []).join(", ")],
      ["Experience", exp.years != null ? `${exp.years} years (${exp.source})` : ""],
      ["Organizations", (entities.organizations || []).join(", ")],
      ["Locations", (entities.locations || []).join(", ")],
      ["Dates", (entities.dates || []).slice(0, 6).join(", ")],
    ];
    return rows.map(([k, v]) => `<tr><th>${esc(k)}</th><td>${v ? esc(v) : '<span class="empty">not found</span>'}</td></tr>`).join("");
  }

  function renderCandidate(c, ner) {
    const s = c.scores;
    const gap = c.skill_gap;
    const t = tone(s.overall);
    const preview = c.extracted_text_preview || "";
    return `
      <article class="card">
        <div class="cand__head">
          <div class="cand__title">
            <div class="rank-badge">#${c.rank}</div>
            <div>
              <h3>${esc((c.entities && c.entities.name) || c.filename)}</h3>
              <div class="rank-sub">${esc(c.filename)} · ${esc(c.word_count)} words · ${esc((c.text_stats || {}).characters || 0)} characters</div>
              <span class="verdict ${t}">${esc(c.verdict)}</span>
            </div>
          </div>
          <div class="gauge" style="--pct:${s.overall};--c:${toneColor(s.overall)}"><div><div><b>${s.overall}</b><br><small>overall</small></div></div></div>
        </div>

        <div class="scores">
          <div class="score"><span>Semantic (MiniLM)</span><b>${s.semantic}</b><small>raw cosine ${s.semantic_raw}%</small></div>
          <div class="score"><span>TF-IDF baseline</span><b>${s.tfidf}</b><small>raw cosine ${s.tfidf_raw}%</small></div>
          <div class="score"><span>Skill coverage</span><b>${s.skill_coverage == null ? "n/a" : s.skill_coverage + "%"}</b><small>${gap.matching.length} of ${gap.required_count} required</small></div>
          <div class="score"><span>Skills found</span><b>${c.skills.length}</b><small>in resume</small></div>
        </div>

        <div class="cols">
          <div>
            <div class="section"><h4>Matched skills (${gap.matching.length})</h4><div class="chips">${chips(gap.matching, "chip--good")}</div></div>
            <div class="section"><h4>Missing required skills (${gap.missing.length})</h4><div class="chips">${chips(gap.missing, "chip--bad", (m) => m.priority === "high" ? " <small>high priority</small>" : "")}</div></div>
            <div class="section"><h4>Additional skills (${gap.extra.length})</h4><div class="chips">${chips(gap.extra, "")}</div></div>
          </div>
          <div>
            <div class="section"><h4>Coverage by skill category</h4>${
              (gap.categories || []).map((cat) => `<div class="cat-row"><span>${esc(cat.category)}</span><div class="bar ${tone(cat.coverage)}"><span style="width:${cat.coverage}%"></span></div><span>${cat.matched}/${cat.required}</span></div>`).join("") || '<p class="empty">No categorised requirements.</p>'
            }</div>
            ${(gap.recommendations || []).length ? `<div class="section"><h4>Suggestions</h4><ul class="tips">${gap.recommendations.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>` : ""}
          </div>
        </div>

        <details class="more" open>
          <summary>NER and extracted candidate information${ner ? "" : " (limited – spaCy NER model not loaded)"}</summary>
          <table class="entity-table"><tbody>${entityRows(c.entities || {})}</tbody></table>
          <div class="section"><h4>Normalised technical skills</h4><div class="chips">${chips(c.skills, "chip--info")}</div></div>
        </details>

        ${preview ? `<details class="more profile-preview"><summary>View extracted resume text (first ${Math.min(preview.length, 1500)} characters)</summary><pre class="pre-block">${esc(preview)}</pre></details>` : ""}

        <div class="actions">
          <button class="btn btn--ghost btn--small ai-from-match" type="button" data-file="${escAttr(c.filename)}">Ask Gemini about this match</button>
        </div>
      </article>`;
  }

  function renderResults(data) {
    lastReport = data;
    const cands = data.candidates || [];
    const scoring = data.scoring || {};
    els.resultsTitle.textContent = "Matching report · " + (els.jobTitle.value.trim() || data.job.title || "Role");
    els.resultsMeta.textContent = `${cands.length} resume${cands.length === 1 ? "" : "s"} ranked · ${data.job.word_count} job words · ${scoring.nlp_backend || "local NLP"}`;

    const warnings = [];
    if (scoring.semantic_is_fallback) warnings.push("Sentence embeddings could not be loaded, so semantic similarity is using the TF-IDF fallback. Raw scores are shown, not invented.");
    if (!scoring.ner_model_loaded) warnings.push("spaCy's en_core_web_sm model is not installed, so organisation/location/date NER is limited.");
    (data.file_errors || []).forEach((e) => warnings.push(`${e.filename}: ${e.error}`));
    els.warn.hidden = !warnings.length;
    els.warn.innerHTML = warnings.map((w) => `<div>${esc(w)}</div>`).join("");

    els.ranking.innerHTML = cands.map((c) => {
      const name = (c.entities && c.entities.name) || c.filename;
      return `<div class="rank-row">
        <div class="rank-badge">${c.rank}</div>
        <div><div class="rank-name">${esc(name)}</div><div class="rank-sub">${esc(c.filename)} · ${esc(c.verdict)}</div>
          <div class="bar ${tone(c.scores.overall)}"><span data-w="${c.scores.overall}"></span></div></div>
        <div class="rank-score">${c.scores.overall}</div></div>`;
    }).join("");

    els.jobSkillCount.textContent = `(${data.job.skills.length} recognised)`;
    els.jobSkills.innerHTML = data.job.skills.length
      ? data.job.skills.map((s) => `<span class="chip chip--info">${esc(s.display)}${s.count > 1 ? ` <small>×${s.count}</small>` : ""}</span>`).join("")
      : '<span class="empty">No recognised technical skills in this description.</span>';

    const labels = { semantic: "Semantic similarity", skills: "Skill coverage", tfidf: "TF-IDF baseline" };
    els.weights.innerHTML = Object.entries(scoring.weights || {}).map(([k, v]) =>
      `<div><span>${labels[k] || esc(k)}</span><b>${Math.round(v * 100)}%</b></div>`).join("");

    els.candidates.innerHTML = cands.map((c) => renderCandidate(c, scoring.ner_model_loaded)).join("");
    setEmpty(els.resultsEmpty, false);
    els.results.hidden = false;
    updateAiResumeOptions(cands[0] && cands[0].filename);
    requestAnimationFrame(() => {
      document.querySelectorAll(".ranking .bar > span").forEach((el) => { el.style.width = el.dataset.w + "%"; });
    });
    loadJobLinks();
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function clearAll() {
    files = [];
    lastReport = null;
    renderFiles();
    clearAlert(els.error);
    els.results.hidden = true;
    els.ranking.innerHTML = "";
    els.candidates.innerHTML = "";
    els.jobLinksCard.hidden = true;
    els.jobLinks.innerHTML = "";
    els.aiResults.innerHTML = "";
    setEmpty(els.resultsEmpty, true);
    setEmpty(els.aiEmpty, true);
  }

  // ------------------------------------------------------- external job links
  async function loadJobLinks() {
    const title = els.jobTitle.value.trim();
    if (!title || els.results.hidden) return;
    try {
      const url = `/api/job-links?title=${encodeURIComponent(title)}&location=${encodeURIComponent(currentLocation())}`;
      const data = await apiFetch(url);
      renderJobLinks(data);
    } catch (e) {
      els.jobLinksCard.hidden = false;
      els.jobLinks.innerHTML = `<p class="alert alert--error">${esc(e.message)}</p>`;
    }
  }
  function scheduleJobLinks() {
    if (els.results.hidden) return;
    clearTimeout(linksTimer);
    linksTimer = setTimeout(loadJobLinks, 250);
  }
  function renderJobLinks(data) {
    els.jobLinksCard.hidden = false;
    els.jobLinksMeta.textContent = `${data.title} · ${data.location}`;
    els.jobLinksDisclaimer.textContent = data.disclaimer;
    els.jobLinks.innerHTML = (data.links || []).map((link) =>
      `<a class="job-link" href="${escAttr(link.url)}" target="_blank" rel="noopener noreferrer">
        <strong>${esc(link.site)}</strong><small>${esc(link.hint)}</small><span>Open search ↗</span>
      </a>`).join("");
  }

  // -------------------------------------------------------------- AI section
  function renderAiStatus(gem) {
    if (!gem || !els.aiStatus) return;
    const configured = !!gem.configured;
    els.aiStatus.className = "status-card " + (configured ? "ok" : "warn");
    els.aiStatus.innerHTML = configured
      ? `<strong>Connected configuration:</strong> ${esc(gem.provider)} · model <code>${esc(gem.model)}</code><br><span class="hint">Cloud requests require explicit consent each time.</span>`
      : `<strong>Gemini not configured.</strong> ${esc(gem.note || "Set GEMINI_API_KEY to enable cloud AI.")}<br><span class="hint">All local matching features still work normally.</span>`;
  }
  async function testAiConnection() {
    clearAlert(els.aiError);
    setButtonBusy(els.aiTestBtn, true, "Testing…");
    try {
      const data = await apiFetch("/api/ai/test", { method: "POST" });
      els.aiStatus.className = "status-card ok";
      els.aiStatus.innerHTML = `<strong>Connection OK</strong> · model <code>${esc(data.model)}</code> · ${esc(data.latency_ms)} ms`;
    } catch (e) {
      els.aiStatus.className = "status-card bad";
      els.aiStatus.innerHTML = `<strong>Connection failed:</strong> ${esc(e.message)}`;
    } finally {
      setButtonBusy(els.aiTestBtn, false, null, "Test connection");
    }
  }
  function aiFile() {
    const name = els.aiResumeSelect.value;
    const selected = files.find((f) => f.name === name);
    return selected || files[0] || null;
  }
  function addJobToForm(form) {
    const title = els.jobTitle.value.trim();
    const description = els.jobText.value.trim();
    if (description.split(/\s+/).filter(Boolean).length < 10) throw new Error("Enter a job description with at least 10 words.");
    form.append("job_title", title || "Custom role");
    form.append("job_description", effectiveJobText());
  }
  async function runAiTask(task, button) {
    clearAlert(els.aiError);
    if (!els.aiConsent.checked) {
      return showAlert(els.aiError, "Explicit consent is required before sending any text to Gemini. Tick the consent checkbox first.");
    }
    const file = aiFile();
    if (!file) return showAlert(els.aiError, "Add a resume first, then choose a task.");
    if (aiBusy) return;

    const form = new FormData();
    form.append("task", task);
    form.append("consent", "true");
    form.append("resume", file, file.name);
    try { addJobToForm(form); } catch (e) { return showAlert(els.aiError, e.message); }

    aiBusy = true;
    setButtonBusy(button, true, "Generating…");
    setEmpty(els.aiEmpty, false);
    els.aiResults.innerHTML = `<article class="card"><p class="hint"><span class="loading-inline" style="border-top-color:var(--brand)"></span> Sending the redacted excerpt and local analysis to Gemini…</p></article>`;
    try {
      const data = await apiFetch("/api/ai/insights", { method: "POST", body: form });
      renderAiResult(data);
    } catch (e) {
      els.aiResults.innerHTML = "";
      setEmpty(els.aiEmpty, true);
      showAlert(els.aiError, e.message);
    } finally {
      aiBusy = false;
      setButtonBusy(button, false, null, "Generate");
    }
  }
  function renderAiResult(data) {
    const r = data.result || {};
    const analysis = data.analysis || {};
    const scores = analysis.scores || {};
    const red = (data.context || {}).redaction || {};
    let body = "";

    if (data.task === "match_explanation") {
      body = `
        <p class="muted-box">${esc(r.summary || "")}</p>
        <div class="grid grid--two">
          <div class="section"><h4>Strengths supported by the analysis</h4><ul class="ai-list">${(r.strengths || []).map((x) => `<li>${esc(x)}</li>`).join("") || '<li class="empty">None</li>'}</ul></div>
          <div class="section"><h4>Gaps and cautions</h4><ul class="ai-list">${(r.gaps || []).map((x) => `<li>${esc(x)}</li>`).join("") || '<li class="empty">None</li>'}</ul></div>
        </div>
        <div class="section"><h4>How to read the score</h4><p>${esc(r.score_interpretation || "")}</p></div>
        <div class="section"><h4>Advisory next steps</h4><ul class="ai-list">${(r.next_steps || []).map((x) => `<li>${esc(x)}</li>`).join("") || '<li class="empty">None</li>'}</ul></div>`;
    } else if (data.task === "resume_improvement") {
      body = `
        <p class="muted-box">${esc(r.summary || "")}</p>
        <div class="subhead">Priority actions</div>
        ${(r.priority_actions || []).map((a) => `<div class="ai-action"><strong>${esc(a.area)}</strong><p>${esc(a.why)}</p><p><b>Action:</b> ${esc(a.action)}</p></div>`).join("") || '<p class="empty">No actions returned.</p>'}
        <div class="grid grid--two">
          <div class="section"><h4>Missing-skill advice</h4><ul class="ai-list">${(r.missing_skill_advice || []).map((x) => `<li>${esc(x)}</li>`).join("") || '<li class="empty">None</li>'}</ul></div>
          <div class="section"><h4>Optional project ideas <span class="hint">(only if genuinely done)</span></h4><ul class="ai-list">${(r.project_ideas || []).map((x) => `<li>${esc(x)}</li>`).join("") || '<li class="empty">None</li>'}</ul></div>
        </div>`;
    } else if (data.task === "bullet_suggestions") {
      body = (r.bullets || []).map((b) => `
        <div class="bullet-pair">
          <div class="bullet bullet--original"><h5>Original</h5>${esc(b.original)}</div>
          <div class="bullet bullet--improved"><h5>Improved</h5>${esc(b.improved)}</div>
        </div>
        <p class="hint" style="margin:-4px 0 12px">Why: ${esc(b.reason)}</p>`).join("") +
        `<div class="section"><h4>Notes</h4><ul class="ai-list">${(r.notes || []).map((x) => `<li>${esc(x)}</li>`).join("") || '<li class="empty">None</li>'}</ul></div>`;
    } else {
      body = `
        <div class="subhead">Technical questions</div>
        ${(r.technical_questions || []).map((q) => `<div class="ai-action"><strong>${esc(q.question)}</strong><p><b>Focus:</b> ${esc(q.focus)}</p>${(q.answer_hints || []).length ? `<p><b>Answer hints:</b> ${esc(q.answer_hints.join(" · "))}</p>` : ""}</div>`).join("") || '<p class="empty">No technical questions returned.</p>'}
        <div class="subhead">HR questions</div>
        ${(r.hr_questions || []).map((q) => `<div class="ai-action"><strong>${esc(q.question)}</strong><p><b>Focus:</b> ${esc(q.focus)}</p></div>`).join("") || '<p class="empty">No HR questions returned.</p>'}
        <div class="section"><h4>Preparation tips</h4><ul class="ai-list">${(r.preparation_tips || []).map((x) => `<li>${esc(x)}</li>`).join("") || '<li class="empty">None</li>'}</ul></div>`;
    }

    const redactionText = `Sent ${(data.context || {}).resume_chars_sent || 0} redacted characters · removed ${red.emails || 0} email(s), ${red.phones || 0} phone(s), ${red.urls || 0} URL(s), ${red.names || 0} name(s)` +
      ((data.context || {}).resume_excerpt_truncated ? " · excerpt truncated" : "");

    els.aiResults.innerHTML = `
      <article class="card">
        <div class="ai-result-head">
          <div><h3>${esc(data.label)}</h3>
          <p class="hint">Model <code>${esc(data.model)}</code> · prompt template <code>${esc(data.template_id)}</code></p></div>
          <div class="chips"><span class="badge badge--cloud">Cloud · Gemini</span><span class="badge badge--local">Scores unchanged</span></div>
        </div>
        <div class="metric-grid">
          ${metric("Local overall", scores.overall == null ? "n/a" : scores.overall, esc(analysis.verdict || ""), "metric--brand")}
          ${metric("Semantic", scores.semantic == null ? "n/a" : scores.semantic, "MiniLM score", "")}
          ${metric("Skill coverage", analysis.skill_coverage == null ? "n/a" : pct(analysis.skill_coverage), "from local gap analysis", "metric--good")}
          ${metric("Missing skills", (analysis.missing_skills || []).length, "from the job", "metric--bad")}
        </div>
        ${body}
        <p class="disclaimer">${esc(redactionText)}. ${esc(data.disclaimer)}</p>
      </article>`;
    setEmpty(els.aiEmpty, false);
    els.aiResults.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function openAiForFile(filename) {
    updateAiResumeOptions(filename);
    els.aiResumeSelect.value = filename;
    clearAlert(els.aiError);
    els.aiResults.innerHTML = `<article class="card"><p class="hint">Ready to analyse <strong>${esc(filename)}</strong>. Tick the consent box and choose a task.</p></article>`;
    setEmpty(els.aiEmpty, false);
    els.aiCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // -------------------------------------------------------------- listeners
  els.jobSelect.addEventListener("change", applyJob);
  els.jobTitle.addEventListener("input", () => { els.jobSelect.value = ""; scheduleJobLinks(); });
  els.jobText.addEventListener("input", () => { updateCounter(); els.jobSelect.value = ""; });
  els.jobLocation.addEventListener("input", scheduleJobLinks);
  document.querySelectorAll("[data-location]").forEach((btn) => {
    btn.addEventListener("click", () => { els.jobLocation.value = btn.dataset.location; scheduleJobLinks(); });
  });

  els.dropzone.addEventListener("click", () => els.fileInput.click());
  els.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); }
  });
  els.fileInput.addEventListener("change", () => { addFiles(els.fileInput.files); els.fileInput.value = ""; });
  ["dragenter", "dragover"].forEach((ev) => els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove("drag"); }));
  els.dropzone.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
  els.fileList.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-remove]");
    if (btn) { files.splice(Number(btn.dataset.remove), 1); renderFiles(); }
  });

  els.analyze.addEventListener("click", analyze);
  els.clear.addEventListener("click", clearAll);
  els.matchingSampleBtn.addEventListener("click", loadSampleResume);
  els.candidates.addEventListener("click", (e) => {
    const btn = e.target.closest(".ai-from-match");
    if (btn) openAiForFile(btn.dataset.file);
  });

  els.aiTestBtn.addEventListener("click", testAiConnection);
  els.aiResumeSelect.addEventListener("change", () => clearAlert(els.aiError));
  els.aiTaskButtons.addEventListener("click", (e) => {
    const btn = e.target.closest(".ai-run");
    if (btn) runAiTask(btn.dataset.task, btn);
  });

  updateCounter();
  setEmpty(els.aiEmpty, true);
  loadStatus();
  loadJobs();
})();
