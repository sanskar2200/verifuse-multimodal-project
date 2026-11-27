"use strict";

const ICONS = {
  scan: '<path d="M8 3H4a1 1 0 0 0-1 1v4m13-5h4a1 1 0 0 1 1 1v4M3 16v4a1 1 0 0 0 1 1h4m8 0h4a1 1 0 0 0 1-1v-4M8 12l3 3 5-6"/>',
  chart: '<path d="M4 4v16h17M8 15v-4m5 4V7m5 8v-6"/>',
  book: '<path d="M12 5v15M3 4c4-1 6 0 9 2 3-2 5-3 9-2v15c-4-1-6 0-9 2-3-2-5-3-9-2V4Z"/>',
  grid: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
  layers: '<path d="m12 3 10 6-10 6L2 9l10-6Zm-10 11 10 6 10-6M2 18l10 6 10-6"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5m0-9v.1"/>',
  upload: '<path d="M12 16V3m-4 4 4-4 4 4M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5"/>',
  image: '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8" cy="8" r="1.5"/><path d="m3 17 5-5 4 4 4-7 5 8"/>',
  text: '<path d="M4 5h16M12 5v15M8 20h8M4 5v3m16-3v3"/>',
  close: '<path d="m6 6 12 12M18 6 6 18"/>',
  lock: '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 5v2"/>',
  route: '<circle cx="5" cy="6" r="2"/><circle cx="19" cy="18" r="2"/><path d="M7 6h8a4 4 0 0 1 0 8H9a2 2 0 0 0 0 4h8"/>',
  fingerprint: '<path d="M5 12v3m-1-7a9 9 0 0 1 16 0m-13 3a5 5 0 0 1 10 0v5m-7 5c2-4 2-7 2-10m-4 8c1-3 1-5 1-8a3 3 0 0 1 6 0c0 3 0 6-1 10m4-2c1-3 1-6 1-9M5 19l1-2"/>',
  shield: '<path d="m12 2 9 4v6c0 5-9 10-9 10S3 17 3 12V6l9-4Z"/><path d="m8 12 3 3 5-6"/>',
  download: '<path d="M12 3v13m-4-4 4 4 4-4M4 17v4h16v-4"/>',
  refresh: '<path d="M20 8a8 8 0 0 0-14-3L3 8m0-5v5h5M4 16a8 8 0 0 0 14 3l3-3m0 5v-5h-5"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  spark: '<path d="m12 3 2 7 7 2-7 2-2 7-2-7-7-2 7-2 2-7Z"/>'
};
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const icon = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ICONS.info}</svg>`;
const pct = (value, decimals = 1) => Number.isFinite(Number(value)) && value !== null && value !== undefined ? `${(Number(value) * 100).toFixed(decimals)}%` : "—";
const clamp = value => Math.max(0, Math.min(1, Number(value) || 0));
const humanize = value => String(value || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
const state = {profile: "synthetic", samples: [], sampleOffset: 0, sampleId: null, imageBase64: null, imageSrc: null, imageName: null, result: null, resultInput: null, experiments: null, busy: false, uploadPending: false, uploadGeneration: 0};
let toastTimer;

$$('[data-icon]').forEach(element => { element.innerHTML = icon(element.dataset.icon); });

async function api(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), options.method === "POST" ? 120000 : 30000);
  try {
    const response = await fetch(path, {...options, signal: controller.signal});
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) throw new Error(response.ok ? "The server returned an unexpected response." : `The local server returned an error (${response.status}).`);
    const data = await response.json();
    if (!response.ok) {
      const detail = data.error || data.detail || data.message;
      throw new Error(typeof detail === "string" ? detail : "The request could not be processed. Please check your input and try again.");
    }
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The local model took too long to respond. Try again after training has finished.");
    if (error instanceof TypeError) throw new Error("Cannot reach the local server. Start the project server, then refresh this page.");
    throw error;
  } finally { clearTimeout(timer); }
}

function toast(message) {
  clearTimeout(toastTimer);
  $('#toast').textContent = message;
  $('#toast').hidden = false;
  toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 4200);
}

function switchView() {
  const requested = location.hash.slice(1) || "verify";
  const name = ["verify", "experiments", "about"].includes(requested) ? requested : "verify";
  $$('.view').forEach(element => { element.hidden = element.id !== `view-${name}`; });
  $$('.nav-link').forEach(element => {
    const selected = element.dataset.view === name;
    element.classList.toggle("active", selected);
    if (selected) element.setAttribute("aria-current", "page"); else element.removeAttribute("aria-current");
  });
  $('#breadcrumb-title').textContent = {verify: "Verify content", experiments: "Experiments", about: "Project notes"}[name];
  document.title = `${{verify:"Verify content", experiments:"Experiments", about:"Project notes"}[name]} · VeriFuse`;
  $('#sidebar').classList.remove("open");
  $('#menu-toggle').setAttribute("aria-expanded", "false");
  window.scrollTo(0, 0);
  if (name === "experiments" && !state.experiments) loadExperiments();
  if (name === "about" && !state.experiments) loadExperiments(false);
}

window.addEventListener("hashchange", switchView);
$('#menu-toggle').addEventListener("click", () => {
  const open = $('#sidebar').classList.toggle("open");
  $('#menu-toggle').setAttribute("aria-expanded", String(open));
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    $('#sidebar').classList.remove("open");
    $('#menu-toggle').setAttribute("aria-expanded", "false");
  }
});
document.addEventListener("click", event => {
  if ($('#sidebar').classList.contains("open") && !event.target.closest("#sidebar, #menu-toggle")) {
    $('#sidebar').classList.remove("open");
    $('#menu-toggle').setAttribute("aria-expanded", "false");
  }
});

async function loadHealth() {
  try {
    const data = await api("/api/health");
    applyProfile(data.profile);
    const ready = Boolean(data.model_ready);
    $('#connection-label').textContent = ready ? `${state.profile === "real" ? "Real-media" : "Synthetic"} model ready` : "Server online · model not trained";
    $('#connection').classList.toggle("offline", !ready);
    $('#connection').title = ready ? "The local model is available for analysis." : "Run the training command from the README to prepare the models.";
  } catch (error) {
    $('#connection-label').textContent = "Local server unavailable";
    $('#connection').classList.add("offline");
    $('#connection').title = error.message;
  }
}

function applyProfile(profile) {
  if (profile !== "real") return;
  state.profile = "real";
  document.documentElement.dataset.profile = "real";
  $('.lab-tag').textContent = "REAL-MEDIA RESEARCH MODEL";
  $('#view-verify .research-banner p').innerHTML = '<strong>Trained on real-media posts.</strong> This model learns Fakeddit dataset labels, including misleading content and satire. Its score does not verify an image or establish factual truth.';
  $('#view-experiments .research-banner p').innerHTML = '<strong>Fakeddit real-media evaluation.</strong> Results use a held-out subset of downloaded posts. The labels are weak labels derived from source communities; this is not a pixel-level deepfake benchmark.';
  $('#content-text').placeholder = "Enter the headline or caption associated with the image…";
  $('#text-help').textContent = "Provide both the post’s text and image. Scores refer to the Fakeddit label classes.";
  $('.model-field > label').textContent = "MODEL APPROACH";
  if (!$('#model-select option[value="text_only"]')) {
    $('#model-select').insertAdjacentHTML('beforeend', '<option value="text_only">Caption-only baseline</option><option value="image_only">Image-only baseline</option>');
  }
  $('.empty-capabilities').lastElementChild.innerHTML = '<i></i>Modality comparison';
  $('#loading-result p').innerHTML = 'Encoding the image and caption,<br>and comparing the model signals.';
  $('.sample-head > span').textContent = "Or inspect a held-out dataset post";
  $('.about-intro > p:nth-of-type(3)').textContent = "The real-media model trains on downloaded Fakeddit posts and images, using pretrained text and vision representations. Training, validation, and test subsets are kept separate. The earlier synthetic shape benchmark is retained as a separate profile.";
  $('.about-pills').innerHTML = '<span>Real-media posts</span><span>Fusion comparison</span><span>Modality evidence</span><span>Held-out evaluation</span>';
  $('.limitations-card > p').textContent = "REAL and FAKE refer to the model’s predicted Fakeddit label. Community-derived labels can include satire, misleading context, and manipulated content; they do not certify individual posts.";
  $('.limitations-card ul').innerHTML = '<li>A natural photograph may have a FAKE dataset label because of its accompanying post or source context.</li><li>Scores are uncalibrated model estimates, not measured probabilities that a claim is true or an image was manipulated.</li><li>The model has no validated out-of-distribution detector. Unfamiliar content can still receive a confident score.</li><li>Modality scores and representation similarity describe model behavior. They are not forensic evidence.</li>';
  const steps = $$('.method-steps > div p');
  steps[0].textContent = "Encode the post text and natural image with pretrained representations.";
  steps[2].textContent = "Compare image and caption classifier scores with their representation similarity.";
  steps[3].textContent = "Select settings on validation data, then report held-out test metrics.";
  $('.dataset-about > p').textContent = "The sample gallery draws from the real-media evaluation split. The collection contains downloaded posts and their dataset labels, with provenance and split details recorded by the training pipeline.";
  const deliverables = $$('.dataset-about')[1];
  deliverables.querySelector('h2').textContent = "Project deliverables";
  deliverables.querySelector('p').textContent = "The paper and slides describe the earlier synthetic benchmark. The project archive includes both profiles and the real-data training report; third-party media and encoder weights are downloaded separately. Current real-media metrics are available in Experiments and its JSON export. Student names and IDs remain editable.";
  const labels = deliverables.querySelectorAll('.deliverable-links small');
  labels[0].textContent = "Synthetic benchmark · editable Word";
  labels[1].textContent = "Synthetic benchmark · PDF";
  labels[2].textContent = "Synthetic benchmark · editable slides";
  labels[3].textContent = "Both profiles · source, classifiers and reports";
  if (state.experiments && state.experiments.profile === "real") renderDatasetSummary(state.experiments.dataset);
}

async function loadSamples() {
  try {
    const data = await api("/api/samples");
    applyProfile(data.profile);
    state.samples = Array.isArray(data.samples) ? data.samples : [];
    renderSamples();
  } catch (error) {
    $('#samples').innerHTML = `<div class="samples-loading">Sample collection is unavailable.<br>${esc(error.message)}</div>`;
    $('#more-samples').disabled = true;
  }
}

function isAltered(sample) {
  return sample.label === 1 || String(sample.label).toUpperCase() === "FAKE";
}

function renderSamples() {
  const shown = state.samples.slice(state.sampleOffset, state.sampleOffset + 4);
  $('#more-samples').disabled = state.busy || state.samples.length <= 4;
  if (!shown.length) {
    $('#samples').innerHTML = '<div class="samples-loading">No samples are available for this model profile. Prepare its dataset using the README.</div>';
    return;
  }
  $('#samples').innerHTML = shown.map(sample => {const label = state.profile === "real" ? `${isAltered(sample) ? "FAKE" : "REAL"}-labeled post` : isAltered(sample) ? "Altered sample" : "Real sample"; return `<button type="button" class="sample-button${String(sample.id) === String(state.sampleId) ? " selected" : ""}" data-sample-id="${esc(sample.id)}" aria-label="Load ${esc(label)}: ${esc(sample.caption || sample.id)}" aria-pressed="${String(sample.id) === String(state.sampleId)}"${state.busy ? " disabled" : ""}><img src="${esc(sample.image_url)}" alt="" loading="lazy"><span class="sample-marker${isAltered(sample) ? " altered" : ""}"></span><span>${esc(label)}</span></button>`;}).join("");
  $$('#samples [data-sample-id]').forEach(button => button.addEventListener("click", () => selectSample(button.dataset.sampleId)));
}

function selectSample(id) {
  if (state.busy) return;
  const sample = state.samples.find(item => String(item.id) === String(id));
  if (!sample) return;
  cancelPendingUpload();
  state.sampleId = sample.id;
  state.imageBase64 = null;
  state.imageSrc = sample.image_url;
  state.imageName = `Lab sample ${sample.id} · ${sample.split || "dataset"}`;
  $('#content-text').value = sample.caption || sample.text || "";
  $('#image-file').value = "";
  updateCount();
  updatePreview();
  renderSamples();
  resetResult();
  clearError();
}

$('#more-samples').addEventListener("click", () => {
  if (!state.samples.length) return;
  state.sampleOffset = state.sampleOffset + 4 >= state.samples.length ? 0 : state.sampleOffset + 4;
  renderSamples();
});

function updateCount() { $('#text-count').textContent = `${$('#content-text').value.length.toLocaleString()} / 3,000`; }
$('#content-text').addEventListener("input", () => { updateCount(); clearError(); resetResult(); });
$('#model-select').addEventListener("change", resetResult);

function updatePreview() {
  const present = Boolean(state.imageSrc);
  $('#upload-empty').hidden = present;
  $('#upload-preview').hidden = !present;
  $('#dropzone').classList.toggle("has-image", present);
  if (present) {
    $('#preview-image').src = state.imageSrc;
    $('#preview-name').textContent = state.imageName || "Selected image";
  } else $('#preview-image').removeAttribute("src");
}

function clearError() { $('#form-error').hidden = true; $('#form-error').textContent = ""; }
function showError(message) { $('#form-error').textContent = message; $('#form-error').hidden = false; }

function cancelPendingUpload() {
  state.uploadGeneration += 1;
  state.uploadPending = false;
  updateAnalyzeButton();
}

function updateAnalyzeButton() {
  $('#analyze-button').disabled = state.busy || state.uploadPending;
  $('#analyze-label').textContent = state.busy ? "Analyzing…" : state.uploadPending ? "Reading image…" : "Analyze content";
}

async function readImage(file) {
  if (!file || state.busy) return;
  cancelPendingUpload();
  const generation = state.uploadGeneration;
  clearError();
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    showError("Choose a JPG, PNG, or WEBP image. Other file types are not supported.");
    $('#image-file').value = "";
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    showError("This image is larger than 8 MB. Choose a smaller file.");
    $('#image-file').value = "";
    return;
  }
  state.uploadPending = true;
  updateAnalyzeButton();
  resetResult();
  try {
    const dataUrl = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("The image could not be read. Choose the file again."));
      reader.readAsDataURL(file);
    });
    if (generation !== state.uploadGeneration || state.busy) return;
    await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = resolve;
      image.onerror = () => reject(new Error("This file does not contain a readable image. Choose another JPG, PNG, or WEBP file."));
      image.src = dataUrl;
    });
    if (generation !== state.uploadGeneration || state.busy) return;
    state.imageBase64 = dataUrl;
    state.imageSrc = dataUrl;
    state.imageName = file.name;
    state.sampleId = null;
    updatePreview();
    renderSamples();
    resetResult();
  } catch (error) {
    if (generation === state.uploadGeneration) showError(error.message);
  } finally {
    if (generation === state.uploadGeneration) {
      state.uploadPending = false;
      updateAnalyzeButton();
    }
  }
}

$('#image-file').addEventListener("change", event => readImage(event.target.files[0]));
$('#remove-image').addEventListener("click", event => {
  event.preventDefault();
  if (state.busy) return;
  cancelPendingUpload();
  state.imageBase64 = null;
  state.imageSrc = null;
  state.imageName = null;
  state.sampleId = null;
  $('#image-file').value = "";
  updatePreview();
  renderSamples();
  resetResult();
});
['dragenter', 'dragover'].forEach(name => $('#dropzone').addEventListener(name, event => {
  event.preventDefault();
  if (!state.busy) $('#dropzone').classList.add("drag-over");
}));
['dragleave', 'drop'].forEach(name => $('#dropzone').addEventListener(name, event => {
  event.preventDefault();
  $('#dropzone').classList.remove("drag-over");
}));
$('#dropzone').addEventListener("drop", event => {
  if (event.dataTransfer.files.length > 1) { showError("Choose one image at a time."); return; }
  readImage(event.dataTransfer.files[0]);
});

function resetResult() {
  if (state.busy) return;
  state.result = null;
  state.resultInput = null;
  $('#result-content').hidden = true;
  $('#empty-result').hidden = false;
  $('#result-status').textContent = "AWAITING ANALYSIS";
}

function setBusy(busy) {
  state.busy = busy;
  updateAnalyzeButton();
  $('#content-text').disabled = busy;
  $('#model-select').disabled = busy;
  $('#image-file').disabled = busy;
  $('#remove-image').disabled = busy;
  $('#more-samples').disabled = busy || state.samples.length <= 4;
  $$('.sample-button').forEach(element => { element.disabled = busy; });
  $('#loading-result').hidden = !busy;
  $('#output-title').closest('section').setAttribute('aria-busy', String(busy));
}

$('#verify-form').addEventListener("submit", async event => {
  event.preventDefault();
  if (state.busy) return;
  if (state.uploadPending) {
    showError("Wait for the selected image to finish loading before analyzing.");
    return;
  }
  clearError();
  const text = $('#content-text').value.trim();
  if (!text && !state.imageSrc) {
    showError("Add a caption, upload an image, or choose a lab sample to begin.");
    $('#content-text').focus();
    return;
  }
  $('#empty-result').hidden = true;
  $('#result-content').hidden = true;
  $('#result-status').textContent = "ANALYZING";
  const inputSnapshot = Object.freeze({text, model:$('#model-select').value, sample_id:state.sampleId, image_base64:state.imageBase64, image_src:state.imageSrc, image_name:state.imageName});
  state.result = null;
  state.resultInput = null;
  setBusy(true);
  try {
    const result = await api("/api/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({text:inputSnapshot.text, image_base64:inputSnapshot.image_base64, sample_id:inputSnapshot.sample_id, model:inputSnapshot.model})});
    state.result = result;
    state.resultInput = inputSnapshot;
    renderResult(result, inputSnapshot);
    $('#result-status').textContent = "ANALYSIS COMPLETE";
    if (window.innerWidth < 990) $('#output-title').scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches ? "auto" : "smooth", block:"start"});
  } catch (error) {
    showError(error.message);
    $('#empty-result').hidden = false;
    $('#result-status').textContent = "ANALYSIS UNAVAILABLE";
  } finally { setBusy(false); }
});

function readable(value) {
  if (typeof value === "string" || typeof value === "number") return String(value);
  if (!value) return "";
  if (Array.isArray(value)) return value.map(readable).join(" · ");
  return value.summary || value.description || value.text || value.detail || Object.entries(value).filter(([,v]) => typeof v !== "object").map(([k,v]) => `${humanize(k)}: ${v}`).join(" · ");
}

function renderResult(result, inputSnapshot) {
  const prediction = ["REAL","FAKE","UNCERTAIN"].includes(String(result.prediction).toUpperCase()) ? String(result.prediction).toUpperCase() : "UNCERTAIN";
  const kind = prediction.toLowerCase();
  const colors = {real:"#97b979",fake:"#d7987d",uncertain:"#d2ba79"};
  const score = result.fake_probability;
  const evidence = Array.isArray(result.evidence) ? result.evidence : [];
  const tokens = Array.isArray(result.token_attributions) ? result.token_attributions : [];
  const regions = Array.isArray(result.image_regions) ? result.image_regions : [];
  const warnings = Array.isArray(result.warnings) ? [...result.warnings] : [];
  if (result.in_domain === false && !warnings.some(item => /distribution|domain|unfamiliar/i.test(readable(item)))) warnings.unshift("This input is outside the controlled training distribution. Treat the score as experimental.");
  const explanation = readable(result.explanation);
  const realProfile = state.profile === "real";
  const domainText = realProfile ? "Generalization to this input is unverified" : result.in_domain === true ? "Within supported scene domain" : result.in_domain === false ? "Outside supported scene domain" : "Domain status unavailable";
  $('#result-content').innerHTML = `<div class="result-summary"><div class="probability-ring" style="--score:${clamp(score)*100}%;--score-color:${colors[kind]}"><div class="ring-value">${esc(pct(score, 0))}<span>${realProfile ? "FAKE CLASS SCORE" : "FAKE PROBABILITY"}</span></div></div><div><div class="assessment-label">${realProfile ? "PREDICTED DATASET CLASS" : "EXPERIMENTAL MODEL ASSESSMENT"}</div><div class="verdict ${kind}">${prediction}</div><div class="result-detail">${esc(domainText)}<br>${esc(humanize(result.model || inputSnapshot.model))}${!realProfile && result.confidence !== null && result.confidence !== undefined ? ` · ${esc(pct(result.confidence,0))} model confidence` : ""}</div></div></div>
    ${explanation ? `<p class="result-explanation">${esc(explanation)}</p>` : ""}
    <div class="modality-scores">${["text","image"].map(type => {const value = result.modality_scores?.[type];return `<div class="modality-score"><div><span>${humanize(type)} signal</span><strong>${esc(pct(value,0))}</strong></div><div class="mini-bar"><i style="width:${clamp(value)*100}%"></i></div><div class="score-caption">${value === null || value === undefined ? "Modality unavailable" : realProfile ? "Uncalibrated FAKE class score" : "Unimodal fake probability"}</div></div>`;}).join("")}</div>
    ${evidence.length ? `<div class="explanation-block"><h3>${icon("layers")}Evidence breakdown</h3>${evidence.map(item => `<div class="evidence-row"><span class="evidence-icon">${icon(item.type === "text" ? "text" : item.type === "image" ? "image" : "spark")}</span><div><h4>${esc(item.title || humanize(item.type) || "Model signal")}</h4><p>${esc(readable(item.detail || item.description))}</p></div></div>`).join("")}</div>` : ""}
    ${tokens.length ? `<div class="explanation-block"><h3>${icon("text")}Words that shift the score</h3><div class="token-cloud">${tokens.slice(0,120).map(item => {const impact = Number(item.impact) || 0; return `<span class="token ${impact > .001 ? "toward-fake" : impact < -.001 ? "toward-real" : ""}" title="Fake-score change: ${impact >= 0 ? "+" : ""}${(impact*100).toFixed(2)} percentage points">${esc(item.token)}</span>`;}).join(" ")}</div><div class="token-legend"><span><i class="legend-square"></i>Toward fake</span><span><i class="legend-square real"></i>Toward real</span></div><p class="explanation-caption">Token removal measures local sensitivity. A highlight is not proof of manipulation.</p></div>` : ""}
    ${regions.length && inputSnapshot.image_src ? `<div class="explanation-block"><h3>${icon("image")}Image regions that shift the score</h3><div class="region-image"><img src="${esc(inputSnapshot.image_src)}" alt="Input image with model-sensitive regions highlighted">${regions.map(region => {const x=clamp(region.x), y=clamp(region.y), width=Math.min(clamp(region.width),1-x), height=Math.min(clamp(region.height),1-y), impact=Number(region.impact)||0;return `<span class="region-overlay ${impact < 0 ? "real" : ""}" style="left:${x*100}%;top:${y*100}%;width:${width*100}%;height:${height*100}%" title="Region score change: ${impact>=0?"+":""}${(impact*100).toFixed(2)} percentage points"></span>`;}).join("")}</div><p class="explanation-caption">Occlusion sensitivity identifies regions that affect this model’s score.</p></div>` : ""}
    ${warnings.length ? `<div class="warnings">${warnings.map(warning => `<p>${esc(readable(warning))}</p>`).join("")}</div>` : ""}
    <div class="result-actions"><span>${Number.isFinite(Number(result.elapsed_ms)) ? `${Math.round(Number(result.elapsed_ms)).toLocaleString()} ms · ` : ""}Local inference</span><button type="button" class="download-button" id="download-result">${icon("download")}Export analysis</button></div>`;
  $('#result-content').hidden = false;
  $('#download-result').addEventListener("click", downloadResult);
}

function downloadJSON(data, filename) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type:"application/json"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadResult() {
  if (!state.result || !state.resultInput) return;
  const input = state.resultInput;
  downloadJSON({exported_at:new Date().toISOString(),project:"VeriFuse",profile:state.profile,note:state.profile === "real" ? "Experimental Fakeddit dataset-class prediction on real media; uncalibrated score, not an authenticity determination." : "Controlled synthetic research demonstration; not an authenticity determination.",input:{text:input.text,sample_id:input.sample_id,image_name:input.image_name,model:input.model},result:state.result},`verifuse-analysis-${Date.now()}.json`);
  toast("Analysis exported as JSON.");
}

$('#refresh-experiments').addEventListener("click", () => loadExperiments());
let experimentsLoading = false;
async function loadExperiments(showLoading = true) {
  if (experimentsLoading) return;
  experimentsLoading = true;
  $('#refresh-experiments').disabled = true;
  if (showLoading) $('#experiments-content').innerHTML = '<div class="card content-empty"><div class="spinner"></div><h3>Loading experiment results…</h3></div>';
  try {
    const data = await api("/api/experiments");
    applyProfile(data.profile);
    state.experiments = data;
    renderExperiments(data);
    renderDatasetSummary(data.dataset);
  } catch (error) {
    $('#experiments-content').innerHTML = `<div class="card content-empty">${icon("chart")}<h3>Experiments are not available yet.</h3><p>${esc(error.message)}</p><p>Train and evaluate the models using the README, then refresh results.</p></div>`;
  } finally { experimentsLoading = false; $('#refresh-experiments').disabled = false; }
}

function metricRows(experiments, bestId, includeMacro = false) {
  const keys = includeMacro ? ["accuracy","precision","recall","f1","macro_f1","roc_auc"] : ["accuracy","precision","recall","f1","roc_auc"];
  return experiments.map(experiment => `<tr class="${experiment.id === bestId ? "best-row" : ""}"><td>${esc(experiment.name || humanize(experiment.id || experiment.model))}</td>${keys.map(key => `<td>${esc(pct(experiment.metrics?.[key]))}</td>`).join("")}</tr>`).join("");
}

function normalizeVariants(experiment) {
  const values = experiment?.variants || experiment?.ablations || experiment?.results || [];
  if (Array.isArray(values)) return values;
  return Object.entries(values).map(([key, value]) => ({id:key,name:humanize(key),...(typeof value === "object" ? value : {f1:value})}));
}

function renderDatasetSummary(dataset) {
  if (!dataset) return;
  const splits = dataset.splits || {};
  $('#dataset-summary').innerHTML = `${dataset.total_records ? `<span><strong>${esc(dataset.total_records)}</strong> ${state.profile === "real" ? "real-media posts" : "generated records"}</span>` : ""}${Object.entries(splits).map(([split, count]) => `<span><strong>${esc(typeof count === "object" ? count.records ?? count.count ?? readable(count) : count)}</strong> ${esc(humanize(split))}</span>`).join("")}`;
}

function matrixHtml(matrix) {
  if (!Array.isArray(matrix) || !Array.isArray(matrix[0]) || !Array.isArray(matrix[1])) return '<p class="experiment-footnote">Confusion matrix is unavailable for this run.</p>';
  return `<div class="confusion-wrap"><span class="matrix-y-label">ACTUAL CLASS</span><div class="confusion-matrix"><span></span><span class="matrix-label">REAL</span><span class="matrix-label">FAKE</span><span class="matrix-label">REAL</span><div class="matrix-cell correct">${esc(matrix[0][0])}<small>TRUE NEGATIVE</small></div><div class="matrix-cell">${esc(matrix[0][1])}<small>FALSE POSITIVE</small></div><span class="matrix-label">FAKE</span><div class="matrix-cell">${esc(matrix[1][0])}<small>FALSE NEGATIVE</small></div><div class="matrix-cell correct">${esc(matrix[1][1])}<small>TRUE POSITIVE</small></div></div></div><div class="matrix-caption">PREDICTED CLASS</div>`;
}

function flattenScalars(value, prefix = "", maxDepth = 2) {
  if (!value || typeof value !== "object" || maxDepth < 0) return [];
  return Object.entries(value).flatMap(([key,item]) => {
    const label = prefix ? `${prefix} · ${humanize(key)}` : humanize(key);
    if (typeof item === "number" || typeof item === "boolean" || (typeof item === "string" && item.length < 200)) return [[label,item]];
    if (item && typeof item === "object" && !Array.isArray(item)) return flattenScalars(item,label,maxDepth-1);
    return [];
  });
}

function renderExperiments(data) {
  const realProfile = data.profile === "real" || state.profile === "real";
  const experiments = Array.isArray(data.experiments) ? data.experiments : [];
  if (!experiments.length) {
    $('#experiments-content').innerHTML = '<div class="card content-empty"><h3>The next step is an experiment.</h3><p>No completed evaluation is available yet.<br>Run training and evaluation from the README, then refresh this page.</p></div>';
    return;
  }
  const fusion = experiments.filter(item => ["feature_fusion","late_fusion"].includes(item.id));
  const best = realProfile ? experiments.find(item => item.id === (data.selected_model || "feature_fusion") || item.model === data.selected_model) : [...(fusion.length ? fusion : experiments.filter(item=>item.metrics))].sort((a,b)=>(Number(b.metrics?.f1)||0)-(Number(a.metrics?.f1)||0))[0];
  const comparisons = experiments.filter(item=>["text_baseline","image_baseline","feature_fusion","late_fusion"].includes(item.id));
  const displayComparisons = comparisons.length ? comparisons : experiments.filter(item=>item.metrics);
  const ablation = experiments.find(item=>item.id === "modality_ablation");
  const ood = experiments.find(item=>item.id === "ood_generalization");
  const fullFusion = experiments.find(item => item.id === "feature_fusion");
  const variants = ablation ? [...(fullFusion ? [{id:"full_model",name:"Full feature fusion",metrics:fullFusion.metrics}] : []), ...normalizeVariants(ablation)] : [];
  const oodVariants = normalizeVariants(ood);
  const hasOodMetrics = Boolean(ood?.metrics || oodVariants.some(item => item.metrics));
  const difficult = Array.isArray(data.difficult_cases) ? data.difficult_cases : [];
  const faithfulness = flattenScalars(data.faithfulness);
  const sanity = !realProfile && Array.isArray(data.sanity_checks) ? data.sanity_checks : [];
  const total = data.dataset?.total_records;
  const splits = data.dataset?.splits || {};
  const testCount = typeof splits.test === "object" ? splits.test?.count ?? splits.test?.records : splits.test;
  const bestName = best?.name || humanize(best?.id);
  const majority = realProfile ? data.majority_baseline : null;
  const runCard = majority ? `<div class="card metric-card"><span>Majority-class accuracy</span><strong>${esc(pct(majority.test_accuracy))}</strong><p>Always predict ${esc(majority.class || "the training majority")} · test baseline</p></div>` : `<div class="card metric-card"><span>Completed experiments</span><strong>${experiments.filter(item => item.metrics || normalizeVariants(item).some(variant => variant.metrics)).length.toString().padStart(2,"0")}</strong><p>Seed ${esc(data.seed ?? "—")} · reproducible evaluation</p></div>`;
  $('#experiments-content').innerHTML = `<div class="metric-grid"><div class="card metric-card"><span>${realProfile ? "Selected model F1" : "Best fusion F1"}</span><strong>${esc(pct(best?.metrics?.f1))}</strong><p>${esc(bestName || "No fusion result")}</p></div><div class="card metric-card"><span>Held-out accuracy</span><strong>${esc(pct(best?.metrics?.accuracy))}</strong><p>${esc(best?.evaluation_split || "test")} split · ${esc(bestName)}</p></div><div class="card metric-card"><span>${realProfile ? "Real-media posts" : "Generated records"}</span><strong>${esc(total ?? "—")}</strong><p>${esc(testCount ?? "—")} in the held-out test split</p></div>${runCard}</div>
    <div class="experiment-grid"><section class="card experiment-card"><div class="card-heading"><h2>How the models compare</h2><span class="small-tag">HELD-OUT TEST</span></div><div class="table-scroll"><table class="metrics-table"><thead><tr><th>MODEL</th><th>ACCURACY</th><th>PRECISION</th><th>RECALL</th><th>F1 (FAKE)</th>${realProfile ? "<th>MACRO F1</th>" : ""}<th>ROC AUC</th></tr></thead><tbody>${metricRows(displayComparisons,best?.id,realProfile)}</tbody></table></div><p class="experiment-footnote">${realProfile ? "FAKE is the positive dataset class. The highlighted model is selected on validation data before test evaluation. These scores measure agreement with weak Fakeddit labels on held-out downloaded posts. Macro F1 gives both classes equal weight; compare accuracy with the majority-class baseline." : "FAKE is the positive class. The highlighted fusion model has the highest test F1. Scores are measured on synthetic examples."}</p></section><section class="card experiment-card"><div class="card-heading"><h2>Where the model gets it wrong</h2><span class="small-tag">CONFUSION MATRIX</span></div>${matrixHtml(best?.metrics?.confusion_matrix)}<p class="experiment-footnote">${esc(bestName)} · ${esc(best?.evaluation_split || "test")} split. Each cell counts evaluated examples.</p></section></div>
    ${variants.length ? `<section class="card experiment-card experiment-wide"><div class="card-heading"><h2>What each modality contributes</h2><span class="small-tag">ABLATION · F1</span></div><div class="bar-chart">${variants.map(item => {const value=item.metrics?.f1 ?? item.f1;return `<div class="ablation-row"><span>${esc(item.name || item.variant || humanize(item.id || item.model))}</span><div class="ablation-track"><div class="ablation-fill" style="width:${clamp(value)*100}%"></div></div><strong>${esc(pct(value))}</strong></div>`;}).join("")}</div><p class="experiment-footnote">${esc(ablation.description || "Compare the full model with each modality removed.")}</p></section>` : ""}
    ${hasOodMetrics ? `<section class="card experiment-card experiment-wide"><div class="card-heading"><h2>Beyond the training distribution</h2><span class="small-tag">OOD GENERALIZATION</span></div><p class="experiment-footnote">${esc(ood.description || "Evaluation on conditions excluded from training.")}</p><div class="table-scroll"><table class="metrics-table"><thead><tr><th>MODEL / CONDITION</th><th>ACCURACY</th><th>PRECISION</th><th>RECALL</th><th>F1</th><th>ROC AUC</th></tr></thead><tbody>${metricRows(oodVariants.length ? oodVariants : [ood])}</tbody></table></div></section>` : realProfile ? '<section class="card experiment-card experiment-wide"><div class="card-heading"><h2>External generalization remains unmeasured</h2><span class="small-tag">LIMITATION</span></div><p class="experiment-footnote">This run evaluates held-out posts from the same downloaded collection. It does not establish accuracy on other datasets, current news, new manipulation techniques, or arbitrary uploads.</p></section>' : ""}
    ${difficult.length ? `<section class="card experiment-card experiment-wide"><div class="card-heading"><h2>A closer look at difficult cases</h2><span class="small-tag">${difficult.length} EVALUATED CASES</span></div><div class="table-scroll"><table class="metrics-table"><thead><tr><th>CASE / REASONING</th><th>EXPECTED</th><th>PREDICTED</th><th>${realProfile ? "FAKE SCORE" : "P(FAKE)"}</th><th>OUTCOME</th></tr></thead><tbody>${difficult.map(item => {const label=item.label === 1 ? "FAKE" : item.label === 0 ? "REAL" : String(item.label || "—").toUpperCase();const prediction=String(item.prediction || "—").toUpperCase();return `<tr><td><details class="case-detail"><summary>${esc(item.id)} · ${esc(humanize(item.manipulation))}</summary><p>${esc(item.text || item.caption || "")}</p><p>${esc(readable(item.explanation))}</p></details></td><td>${esc(label)}</td><td><span class="case-status ${esc(prediction.toLowerCase())}">${esc(prediction)}</span></td><td>${esc(pct(item.fake_probability))}</td><td>${item.correct === true ? "Correct" : item.correct === false ? "Incorrect" : "—"}</td></tr>`;}).join("")}</tbody></table></div><p class="experiment-footnote">Open a case to inspect its caption and explanation. Correctness is measured against the ${realProfile ? "weak Fakeddit" : "generated"} dataset label.</p></section>` : ""}
    ${faithfulness.length ? `<section class="card experiment-card experiment-wide"><div class="card-heading"><h2>Do the explanations reflect model behavior?</h2><span class="small-tag">EXPLANATION EVALUATION</span></div>${data.faithfulness?.protocol ? `<p class="faithfulness-protocol">${esc(data.faithfulness.protocol)}</p>` : ""}<div class="table-scroll"><table class="metrics-table"><tbody>${faithfulness.map(([key,value])=>`<tr><td>${esc(key)}</td><td>${esc(typeof value === "number" ? Number.isInteger(value) ? value : value.toFixed(4) : value)}</td></tr>`).join("")}</tbody></table></div><p class="experiment-footnote">These checks measure local model behavior. They do not establish that an explanation is causally correct in the real world.</p></section>` : ""}
    ${sanity.length ? `<section class="card experiment-card experiment-wide"><div class="card-heading"><h2>When the model should abstain</h2><span class="small-tag">${sanity.filter(item=>item.passed).length} / ${sanity.length} CHECKS PASSED</span></div><div class="table-scroll"><table class="metrics-table"><thead><tr><th>INPUT CONDITION</th><th>EXPECTED</th><th>ACTUAL</th><th>CHECK</th></tr></thead><tbody>${sanity.map(item=>`<tr><td>${esc(humanize(item.id))}</td><td>${esc(item.expected)}</td><td><span class="case-status ${esc(String(item.actual).toLowerCase())}">${esc(item.actual)}</span></td><td>${item.passed ? "Passed" : "Failed"}</td></tr>`).join("")}</tbody></table></div><p class="experiment-footnote">These checks test explicit domain limits, including missing inputs and unsupported general claims.</p></section>` : ""}
    <section class="card experiment-card experiment-wide"><div class="card-heading"><h2>The complete evaluation record</h2><button class="download-button" id="download-experiments">${icon("download")}Export results</button></div><p class="experiment-footnote">Includes all experiment variants, per-class metrics, explanation evaluations, dataset notes, and sanity checks reported by this run.</p><details class="raw-results"><summary>Inspect run data</summary><pre>${esc(JSON.stringify(data,null,2))}</pre></details></section><div class="experiment-meta"><span>Project: ${esc(readable(data.project) || "VeriFuse")}</span><span>Seed: ${esc(data.seed ?? "—")}</span><span>Loaded from the local evaluation API</span></div>`;
  $('#download-experiments').addEventListener("click", () => { downloadJSON(data,`verifuse-experiments-${Date.now()}.json`); toast("Experiment results exported as JSON."); });
}

switchView();
loadHealth();
loadSamples();

