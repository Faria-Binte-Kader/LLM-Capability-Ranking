// File: llm-tasks.js

// ─── Capability colors ────────────────────────────────────────────────────────
const CAP_COLORS = {
  'Reasoning':                         '#2b6cb0',
  'Narrative Understanding':           '#276749',
  'Social-Aware Communication':        '#97266d',
  'Creativity':                        '#6b46c1',
  'Ethical / Fair Response Generation':'#c53030',
  'Adherence to Instructions':         '#2c7a7b',
  'Multilingual Generation':           '#744210',
  'Personalization & Adaptation':      '#553c9a',
  'Decision Support':                  '#1a365d',
  'Automation & Planning':             '#22543d',
  'Data Understanding':                '#702459',
  'Interaction & Assistance':          '#7b341e',
  'General Cognitive Skills':          '#1a202c',
};

const CAP_ORDER_PREFERRED = [
  'Reasoning', 'Narrative Understanding', 'Social-Aware Communication', 'Creativity',
  'Ethical / Fair Response Generation', 'Adherence to Instructions', 'Multilingual Generation',
  'Personalization & Adaptation', 'Decision Support', 'Automation & Planning',
  'Data Understanding', 'Interaction & Assistance', 'General Cognitive Skills',
];

// ─── Display name helpers ─────────────────────────────────────────────────────
const ORG_NAMES = {
  'qwen':'Qwen','allenai':'AllenAI','deepcogito':'DeepCogito','deepseek-ai':'DeepSeek',
  'google':'Google','meta-llama':'Meta','microsoft':'Microsoft','mistralai':'Mistral AI','tiiuae':'TII UAE',
};

const PREFIX_DISPLAY = {
  'bbq':'BBQ','mmlu':'MMLU','em':'Entity Matching','disinformation':'Disinformation',
};

function titleCase(str) {
  return str.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatSubPart(sub) {
  const hasMixedCase = /[A-Z]/.test(sub) && /[A-Z].*[a-z]/.test(sub);
  return hasMixedCase ? sub.replace(/_/g, ' ') : titleCase(sub);
}

function formatDatasetName(raw) {
  if (datasetDisplayNames[raw]) return datasetDisplayNames[raw];
  if (raw.includes(':')) {
    const colon = raw.indexOf(':'), prefix = raw.slice(0, colon), sub = raw.slice(colon + 1);
    return `${PREFIX_DISPLAY[prefix] || titleCase(prefix)}: ${formatSubPart(sub)}`;
  }
  return titleCase(raw);
}

function formatGroupName(prefix) { return PREFIX_DISPLAY[prefix] || titleCase(prefix); }

function formatModelName(rawId) {
  const idx = rawId.indexOf('_');
  const orgKey = (idx === -1 ? rawId : rawId.slice(0, idx)).toLowerCase();
  const model  = idx === -1 ? rawId : rawId.slice(idx + 1);
  const company = ORG_NAMES[orgKey] || (orgKey.charAt(0).toUpperCase() + orgKey.slice(1));
  let m = model
    .replace(/-instruct(?=-\d{4}$)/i, ' Instruct').replace(/-(\d{4})$/, ' $1')
    .replace(/-instruct$/i, ' Instruct').replace(/-preview$/i, ' Preview')
    .replace(/-it$/i, ' IT').replace(/(\d+)b(?=[-\s]|$)/gi, (_, n) => n + 'B');
  return `${m.charAt(0).toUpperCase() + m.slice(1)} (${company})`;
}

// ─── CSV parser ───────────────────────────────────────────────────────────────
function parseCSV(text) {
  const rows = [];
  let i = 0, n = text.length;
  while (i < n) {
    while (i < n && (text[i] === '\r' || text[i] === '\n')) i++;
    if (i >= n) break;
    const row = [];
    while (i < n && text[i] !== '\n' && text[i] !== '\r') {
      let field = '';
      if (text[i] === '"') {
        i++;
        while (i < n) {
          if (text[i] === '"') { if (i+1<n && text[i+1]==='"') { field+='"'; i+=2; } else { i++; break; } }
          else field += text[i++];
        }
      } else {
        while (i < n && text[i] !== ',' && text[i] !== '\n' && text[i] !== '\r') field += text[i++];
      }
      row.push(field.trim());
      if (i < n && text[i] === ',') i++;
    }
    while (i < n && (text[i] === '\n' || text[i] === '\r')) i++;
    if (row.some(f => f !== '')) rows.push(row);
  }
  return rows;
}

function parseCSVWithHeaders(text) {
  const rows = parseCSV(text);
  if (!rows.length) return [];
  const headers = rows[0];
  return rows.slice(1).map(row => {
    const obj = {};
    headers.forEach((h, j) => { obj[h] = row[j] || ''; });
    return obj;
  });
}

// ─── Dataset name normalisation ───────────────────────────────────────────────
const DISPLAY_NAME_MAP = {
  'BoolQ':'boolq','NarrativeQA':'narrative_qa','TruthfulQA':'truthful_qa',
  'CNN/DailyMail':'cnndm','XSUM':'xsum','IMDB':'imdb','CivilComments':'civil_comments',
  'GSM8K':'gsm8k','HumanEval':'humaneval','LegalSupport':'legal_support','MedQA':'med_qa',
};
const GROUP_PREFIX_MAP = {
  'BBQ':'bbq:','EntityMatching':'em:','Disinformation (climate & covid)':'disinformation:',
};

function resolveDatasetName(csvName, allResultKeys) {
  if (allResultKeys.has(csvName)) return [csvName];
  const mapped = DISPLAY_NAME_MAP[csvName];
  if (mapped) return allResultKeys.has(mapped) ? [mapped] : [];
  const pfx = GROUP_PREFIX_MAP[csvName];
  if (pfx) return Array.from(allResultKeys).filter(k => k.startsWith(pfx));
  const mmluKey = 'mmlu:' + csvName;
  if (allResultKeys.has(mmluKey)) return [mmluKey];
  const emKey = 'em:' + csvName;
  if (allResultKeys.has(emKey)) return [emKey];
  const norm = csvName.toLowerCase().replace(/[\s-]+/g, '_');
  if (allResultKeys.has(norm)) return [norm];
  return [];
}

// ─── State ────────────────────────────────────────────────────────────────────
let llmData = {}, capMap = {}, skillDefs = {}, datasetDescriptions = {};
let metricMeta = {};          // metric_key → { label, explanation, lowerIsBetter }
let datasetDisplayNames = {}; // result_key  → display name (built from CSV)
let validDatasets = new Set(), capOrderFinal = [];
let selectedTemperature = '0.2', rankUpdateTimer = null;

// ─── Init ─────────────────────────────────────────────────────────────────────
Promise.all([
  fetch('results.json').then(r => r.json()),
  fetch('dataset_skill_mapping_updated.csv').then(r => r.text()),
  fetch('skill_definitions.csv').then(r => r.text()),
  fetch('metrics.csv').then(r => r.text()),
]).then(([results, mappingCSV, defsCSV, metricsCSV]) => {
  llmData = results;
  buildValidDatasets();
  buildSkillDefinitions(defsCSV);
  buildMetricMeta(metricsCSV);
  buildCapabilityMap(mappingCSV);
  initTooltip();
  initModal();
  buildCapabilityPanel();
  initTemperatureSelector();
  initCheckboxHandlers();
  initGlobalToggleButtons();
  updateRankingTable();
});

// ─── Data preparation ─────────────────────────────────────────────────────────
function buildValidDatasets() {
  const models = Object.keys(llmData);
  const allDs = new Set();
  for (const m of models) for (const t of Object.values(llmData[m])) for (const d of Object.keys(t)) allDs.add(d);
  for (const ds of allDs) {
    let count = 0;
    for (const m of models) for (const t of Object.values(llmData[m])) if (t[ds]) { count++; break; }
    if (count === models.length) validDatasets.add(ds);
  }
}

function buildSkillDefinitions(csvText) {
  skillDefs = {};
  for (const row of parseCSVWithHeaders(csvText)) {
    const skill = (row['Skill / Certificate'] || '').trim();
    const def   = (row['Definition'] || '').trim();
    const ex    = (row['Example Task / Test Case'] || '').trim();
    if (skill && def) skillDefs[skill] = { def, example: ex };
  }
}

function buildMetricMeta(csvText) {
  metricMeta = {};
  for (const row of parseCSVWithHeaders(csvText)) {
    const key = (row['metric_key'] || '').trim();
    if (!key) continue;
    metricMeta[key] = {
      label:        (row['display_label']  || '').trim() || titleCase(key),
      explanation:  (row['explanation']    || '').trim(),
      lowerIsBetter: (row['lower_is_better'] || '').trim().toLowerCase() === 'true',
    };
  }
}

function buildCapabilityMap(csvText) {
  capMap = {}; datasetDescriptions = {};
  for (const row of parseCSVWithHeaders(csvText)) {
    const csvName = (row['Dataset'] || '').trim();
    const cap     = (row['Capability'] || '').trim();
    const skill   = (row['Skill'] || row['Skill / Certificate'] || '').trim();
    const whatIt  = (row['What It Evaluates'] || '').trim();
    if (!csvName) continue;

    const keys = resolveDatasetName(csvName, validDatasets);
    // Capture display name: CSV "Dataset" column is the human-readable name
    for (const k of keys) if (!datasetDisplayNames[k]) datasetDisplayNames[k] = csvName;
    if (whatIt) for (const k of keys) if (!datasetDescriptions[k]) datasetDescriptions[k] = whatIt;
    if (!cap || !skill || !keys.length) continue;

    if (!capMap[cap]) capMap[cap] = {};
    if (!capMap[cap][skill]) capMap[cap][skill] = new Set();
    for (const k of keys) capMap[cap][skill].add(k);
  }
  for (const cap of Object.keys(capMap)) {
    for (const skill of Object.keys(capMap[cap])) {
      const arr = Array.from(capMap[cap][skill]).sort();
      if (arr.length) capMap[cap][skill] = arr; else delete capMap[cap][skill];
    }
    if (!Object.keys(capMap[cap]).length) delete capMap[cap];
  }
  const seen = new Set();
  capOrderFinal = [];
  for (const c of CAP_ORDER_PREFERRED) if (capMap[c]) { capOrderFinal.push(c); seen.add(c); }
  for (const c of Object.keys(capMap)) if (!seen.has(c)) capOrderFinal.push(c);
}

function getDatasetDescription(key) {
  if (datasetDescriptions[key]) return datasetDescriptions[key];
  if (key.startsWith('mmlu:')) return `Multiple-choice questions testing knowledge in ${key.slice(5).replace(/_/g, ' ')}.`;
  if (key.startsWith('bbq:'))  return `Tests for stereotyping and identity bias related to ${key.slice(4).replace(/_/g, ' ')} in question answering.`;
  if (key.startsWith('em:'))   return `Entity matching benchmark — determines whether two differently-formatted records refer to the same real-world entity.`;
  if (key.startsWith('disinformation:')) return `Tests the model's resistance to ${key.split(':')[1]}-related disinformation and false premise manipulation.`;
  return '';
}

// ─── Score helpers ────────────────────────────────────────────────────────────
function scoreColor(pct) {
  if (pct >= 70) return '#276749';
  if (pct >= 50) return '#b7791f';
  return '#c53030';
}
function scoreBg(pct) {
  if (pct >= 70) return '#f0fff4';
  if (pct >= 50) return '#fffff0';
  return '#fff5f5';
}

// ─── Tooltip (skill hover in left panel) ─────────────────────────────────────
function initTooltip() {
  const tt = document.createElement('div');
  tt.id = 'skill-tooltip';
  tt.style.cssText = 'position:fixed;background:#2d3748;color:#fff;padding:14px 18px;border-radius:10px;font-size:12px;max-width:340px;z-index:9999;display:none;pointer-events:none;line-height:1.6;box-shadow:0 6px 24px rgba(0,0,0,0.35);';
  document.body.appendChild(tt);
  document.addEventListener('mousemove', e => {
    if (tt.style.display === 'none') return;
    const x = e.clientX + 18, y = e.clientY + 12;
    tt.style.left = Math.min(x, window.innerWidth  - tt.offsetWidth  - 12) + 'px';
    tt.style.top  = Math.min(y, window.innerHeight - tt.offsetHeight - 12) + 'px';
  });
}

function showSkillTooltip(skill, datasets) {
  const tt = document.getElementById('skill-tooltip');
  const def = skillDefs[skill];
  const defHtml = def
    ? `<span style="color:#e2e8f0;">${def.def}</span>
       <span style="display:block;margin-top:6px;color:#a0aec0;font-style:italic;">Example: ${def.example}</span>`
    : '';
  const dsNames = datasets.map(d => formatDatasetName(d)).join(', ');
  tt.innerHTML = `<strong style="font-size:13px;display:block;margin-bottom:8px;">${skill}</strong>
    ${defHtml}
    <span style="display:block;margin-top:10px;color:#90cdf4;"><strong style="color:#bee3f8;">Datasets:</strong> ${dsNames}</span>`;
  tt.style.display = 'block';
}

function showScoreTooltip() {
  const tt = document.getElementById('skill-tooltip');
  tt.innerHTML = `<span style="color:#90cdf4;">&#128202; Click to see the full score breakdown</span>`;
  tt.style.display = 'block';
}

function hideTooltip() {
  const tt = document.getElementById('skill-tooltip');
  if (tt) tt.style.display = 'none';
}

// ─── Modal ────────────────────────────────────────────────────────────────────
function initModal() {
  const overlay = document.createElement('div');
  overlay.id = 'skill-modal-overlay';
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.55);z-index:10000;display:none;align-items:center;justify-content:center;padding:20px;';
  overlay.innerHTML = `
    <div id="skill-modal" style="background:#fff;border-radius:14px;max-width:600px;width:100%;max-height:88vh;overflow-y:auto;box-shadow:0 20px 60px rgba(0,0,0,0.3);position:relative;">
      <div id="skill-modal-content" style="padding:28px 32px 32px;"></div>
    </div>
  `;
  document.body.appendChild(overlay);
  overlay.addEventListener('click', e => { if (e.target === overlay) closeModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
}

function openSkillModal(modelName, cap, skill) {
  const modelData = llmData[modelName];
  const datasets  = capMap[cap]?.[skill] || [];
  const def       = skillDefs[skill];
  const capColor  = CAP_COLORS[cap] || '#555';

  let html = `
    <button onclick="closeModal()" style="position:absolute;top:16px;right:18px;background:none;border:none;font-size:22px;cursor:pointer;color:#718096;line-height:1;" title="Close">&#10005;</button>
    <div style="margin-bottom:20px;">
      <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:${capColor};margin-bottom:4px;">${cap}</div>
      <h2 style="margin:0 0 8px;font-size:20px;color:#1a202c;">${skill}</h2>
      ${def ? `<p style="margin:0 0 6px;font-size:13px;color:#4a5568;line-height:1.6;">${def.def}</p>
               <p style="margin:0;font-size:12px;color:#718096;font-style:italic;">Example: ${def.example}</p>` : ''}
    </div>
    <div style="font-size:12px;color:#718096;margin-bottom:20px;padding:10px 14px;background:#f7fafc;border-radius:8px;border-left:3px solid ${capColor};">
      Evaluated for: <strong style="color:#2d3748;">${formatModelName(modelName)}</strong>
    </div>
    <div style="font-size:13px;font-weight:700;color:#2d3748;margin-bottom:14px;text-transform:uppercase;letter-spacing:0.06em;">
      Dataset Breakdown <span style="font-weight:400;color:#a0aec0;font-size:11px;">(${datasets.length} dataset${datasets.length !== 1 ? 's' : ''})</span>
    </div>
  `;

  for (const ds of datasets) {
    const metrics   = getDatasetMetrics(modelData, ds);
    const dsScore   = getDatasetScore(modelData, ds, selectedTemperature);
    const dsPct     = dsScore !== null ? +(dsScore * 100).toFixed(1) : null;
    const dsName    = formatDatasetName(ds);
    const dsDesc    = getDatasetDescription(ds);
    const dsColor   = dsPct !== null ? scoreColor(dsPct) : '#a0aec0';
    const dsBg      = dsPct !== null ? scoreBg(dsPct)    : '#f7fafc';

    html += `
      <div style="margin-bottom:14px;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;">
        <div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;background:${dsBg};border-bottom:1px solid #e2e8f0;">
          <span style="font-weight:600;font-size:14px;color:#1a202c;">${dsName}</span>
          ${dsPct !== null
            ? `<span style="font-size:15px;font-weight:700;color:${dsColor};">${dsPct}%</span>`
            : `<span style="font-size:13px;color:#a0aec0;">N/A</span>`}
        </div>
        ${dsDesc ? `<div style="padding:8px 16px 6px;font-size:12px;color:#718096;line-height:1.5;border-bottom:1px solid #f0f0f0;">${dsDesc}</div>` : ''}
        <div style="padding:10px 16px 14px;">
    `;

    for (const [metric, temps] of Object.entries(metrics)) {
      const rawVal      = temps[selectedTemperature];
      const dispVal     = rawVal != null ? normaliseMetricScore(metric, rawVal) : null;
      const pct         = dispVal !== null ? +(dispVal * 100).toFixed(1) : null;
      const mColor      = pct !== null ? scoreColor(pct) : '#a0aec0';
      const meta        = metricMeta[metricKey(metric)];
      const metricLabel = meta?.label       || titleCase(metric).replace(/\(offline\)/i, '').trim();
      const mExpl       = meta?.explanation || '';

      html += `
        <div style="margin-top:10px;">
          <div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:3px;">
            <span style="font-size:12px;font-weight:600;color:#2d3748;">${metricLabel}</span>
            <span style="font-size:13px;font-weight:700;color:${mColor};">${pct !== null ? pct + '%' : 'N/A'}</span>
          </div>
          ${pct !== null ? `
          <div style="background:#e2e8f0;border-radius:4px;height:6px;margin-bottom:5px;">
            <div style="background:${mColor};width:${Math.min(pct,100)}%;height:6px;border-radius:4px;"></div>
          </div>` : ''}
          ${mExpl ? `<div style="font-size:11px;color:#718096;line-height:1.4;">${mExpl}</div>` : ''}
        </div>
      `;
    }

    html += `</div></div>`;
  }

  document.getElementById('skill-modal-content').innerHTML = html;
  const overlay = document.getElementById('skill-modal-overlay');
  overlay.style.display = 'flex';
}

function closeModal() {
  document.getElementById('skill-modal-overlay').style.display = 'none';
}

// ─── Left panel ───────────────────────────────────────────────────────────────
function buildCapabilityPanel() {
  const container = document.getElementById('task-list');
  container.innerHTML = '';

  for (const cap of capOrderFinal) {
    if (!capMap[cap] || !Object.keys(capMap[cap]).length) continue;
    const color  = CAP_COLORS[cap] || '#555';
    const skills = Object.keys(capMap[cap]);

    const capGroup = document.createElement('div');
    capGroup.className = 'task-group';
    capGroup.style.marginBottom = '10px';

    const header = document.createElement('div');
    header.className = 'task-header';
    header.style.cssText = 'display:flex;align-items:center;gap:8px;padding:4px 0;';

    const capCb = document.createElement('input');
    capCb.type = 'checkbox'; capCb.className = 'cap-cb'; capCb.dataset.cap = cap;
    capCb.checked = true; capCb.style.cssText = 'cursor:pointer;flex-shrink:0;';

    const arrow = document.createElement('span');
    arrow.className = 'cap-arrow'; arrow.innerHTML = '&#9658;';
    arrow.style.cssText = `color:${color};font-weight:bold;width:16px;display:inline-block;flex-shrink:0;cursor:pointer;`;

    const capLabel = document.createElement('span');
    capLabel.style.cssText = `font-weight:bold;font-size:14px;flex:1;cursor:pointer;color:${color};`;
    capLabel.textContent = cap;

    header.appendChild(capCb); header.appendChild(arrow); header.appendChild(capLabel);

    const skillList = document.createElement('div');
    skillList.className = 'cap-skills';
    skillList.style.cssText = 'display:none;padding-left:24px;margin-top:4px;';

    for (const skill of skills) {
      const datasets = capMap[cap][skill];
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;padding:3px 4px 3px 4px;gap:6px;cursor:default;';

      const cb = document.createElement('input');
      cb.type = 'checkbox'; cb.className = 'skill-cb';
      cb.dataset.cap = cap; cb.dataset.skill = skill;
      cb.checked = true; cb.style.cssText = 'cursor:pointer;flex-shrink:0;';

      const lbl = document.createElement('label');
      lbl.style.cssText = 'font-size:13px;cursor:pointer;margin:0;flex:1;';
      lbl.innerHTML = `${skill} <small style="color:#a0aec0;">(${datasets.length} dataset${datasets.length !== 1 ? 's' : ''})</small>`;

      row.appendChild(cb); row.appendChild(lbl);
      row.addEventListener('mouseenter', () => showSkillTooltip(skill, datasets));
      row.addEventListener('mouseleave', hideTooltip);
      skillList.appendChild(row);
    }

    capGroup.appendChild(header); capGroup.appendChild(skillList);
    container.appendChild(capGroup);

    const toggleExpand = () => {
      const expanded = skillList.style.display !== 'none';
      skillList.style.display = expanded ? 'none' : 'block';
      arrow.innerHTML = expanded ? '&#9658;' : '&#9660;';
    };
    arrow.addEventListener('click', toggleExpand);
    capLabel.addEventListener('click', toggleExpand);
    capCb.addEventListener('change', () => {
      skillList.querySelectorAll('.skill-cb').forEach(c => (c.checked = capCb.checked));
      capCb.indeterminate = false; scheduleRankingUpdate();
    });
  }
}

function updateCapCheckboxState(cap) {
  const capCb    = Array.from(document.querySelectorAll('.cap-cb')).find(c => c.dataset.cap === cap);
  if (!capCb) return;
  const skillCbs = Array.from(document.querySelectorAll('.skill-cb')).filter(c => c.dataset.cap === cap);
  const checked  = skillCbs.filter(c => c.checked).length;
  capCb.indeterminate = checked > 0 && checked < skillCbs.length;
  capCb.checked = checked > 0;
}

// ─── Init helpers ─────────────────────────────────────────────────────────────
function initTemperatureSelector() {
  const sel = document.getElementById('temp');
  if (!sel) return;
  selectedTemperature = sel.value;
  sel.addEventListener('change', () => { selectedTemperature = sel.value; scheduleRankingUpdate(); });
}

function initCheckboxHandlers() {
  document.getElementById('task-list').addEventListener('change', e => {
    if (e.target.classList.contains('skill-cb')) {
      updateCapCheckboxState(e.target.dataset.cap); scheduleRankingUpdate();
    }
  });
}

function initGlobalToggleButtons() {
  document.getElementById('all-on')?.addEventListener('click', () => {
    document.querySelectorAll('.skill-cb').forEach(c => (c.checked = true));
    document.querySelectorAll('.cap-cb').forEach(c => { c.checked = true; c.indeterminate = false; });
    scheduleRankingUpdate();
  });
  document.getElementById('all-off')?.addEventListener('click', () => {
    document.querySelectorAll('.skill-cb').forEach(c => (c.checked = false));
    document.querySelectorAll('.cap-cb').forEach(c => { c.checked = false; c.indeterminate = false; });
    scheduleRankingUpdate();
  });
}

// ─── Scoring helpers ──────────────────────────────────────────────────────────
function metricKey(raw) {
  return metricMeta[raw] ? raw : raw.replace(/\s*\(offline\)\s*$/i, '').trim();
}

function normaliseMetricScore(metric, value) {
  return metricMeta[metricKey(metric)]?.lowerIsBetter ? 1 - value : value;
}

function getDatasetScore(modelData, dataset, temp) {
  for (const t of Object.values(modelData))
    if (t[dataset]) {
      const scores = Object.entries(t[dataset])
        .map(([metric, temps]) => {
          const v = temps[temp];
          return v != null ? normaliseMetricScore(metric, v) : null;
        })
        .filter(v => v != null);
      if (scores.length) return scores.reduce((a, b) => a + b, 0) / scores.length;
    }
  return null;
}

function getDatasetMetrics(modelData, dataset) {
  for (const t of Object.values(modelData)) if (t[dataset]) return t[dataset];
  return {};
}

function getSelectedDatasets() {
  const seen = new Set();
  document.querySelectorAll('.skill-cb:checked').forEach(cb => {
    for (const ds of (capMap[cb.dataset.cap]?.[cb.dataset.skill] || [])) seen.add(ds);
  });
  return Array.from(seen);
}

function getSelectedCapSkillDatasets() {
  const checked = new Set();
  document.querySelectorAll('.skill-cb:checked').forEach(cb => checked.add(`${cb.dataset.cap}||${cb.dataset.skill}`));
  const result = [];
  for (const cap of capOrderFinal)
    if (capMap[cap])
      for (const skill of Object.keys(capMap[cap]))
        if (checked.has(`${cap}||${skill}`)) result.push({ cap, skill, datasets: capMap[cap][skill] });
  return result;
}

// ─── Ranking table ────────────────────────────────────────────────────────────
function scheduleRankingUpdate() {
  clearTimeout(rankUpdateTimer);
  rankUpdateTimer = setTimeout(updateRankingTable, 180);
}

function updateRankingTable() {
  const selected = getSelectedDatasets();
  if (!selected.length) { showEmptyMessage(); return; }
  renderRankingTable(selected);
}

function showEmptyMessage() {
  document.querySelector('#llm-table-body').innerHTML = `
    <tr><td colspan="3" style="text-align:center;padding:20px;font-style:italic;">
      Please select at least one skill to rank models.
    </td></tr>`;
}

function renderRankingTable(selectedDatasets) {
  const tableBody         = document.querySelector('#llm-table-body');
  const selectedCapSkills = getSelectedCapSkillDatasets();

  const rows = [];
  for (const [modelName, modelData] of Object.entries(llmData)) {
    let total = 0, count = 0;
    for (const ds of selectedDatasets) {
      const s = getDatasetScore(modelData, ds, selectedTemperature);
      if (s !== null) { total += s; count++; }
    }
    if (count) rows.push({ name: modelName, score: total });
  }
  rows.sort((a, b) => b.score - a.score);

  const htmlParts = [];

  rows.forEach((row, index) => {
    const detailId = `llm-detail-${index + 1}`;
    const byCap = {}, capOrder = [];
    for (const { cap, skill, datasets } of selectedCapSkills) {
      if (!byCap[cap]) { byCap[cap] = []; capOrder.push(cap); }
      byCap[cap].push({ skill, datasets });
    }

    const detailParts = [];
    for (const cap of capOrder) {
      const capColor = CAP_COLORS[cap] || '#555';
      detailParts.push(`
        <tr>
          <td></td>
          <td colspan="2" style="font-size:12px;font-weight:bold;color:${capColor};padding-top:10px;padding-bottom:2px;border-top:1px solid #e2e8f0;">${cap}</td>
          <td></td>
        </tr>
      `);

      for (const { skill, datasets } of byCap[cap]) {
        let skillSum = 0, skillCount = 0;
        for (const ds of datasets) {
          const s = getDatasetScore(llmData[row.name], ds, selectedTemperature);
          if (s !== null) { skillSum += s; skillCount++; }
        }
        const scoreStr = skillCount > 0 ? skillSum.toFixed(3) : 'N/A';

        // Encode model/cap/skill safely as base64 to avoid HTML attribute issues
        const modelEnc = encodeURIComponent(row.name);
        const capEnc   = encodeURIComponent(cap);
        const skillEnc = encodeURIComponent(skill);

        detailParts.push(`
          <tr style="background:#f7fafc;">
            <td></td>
            <td colspan="2" style="font-size:12px;font-weight:600;color:#2d3748;padding-left:12px;padding-top:6px;padding-bottom:6px;">${skill}</td>
            <td style="text-align:right;padding-right:12px;">
              <span class="skill-score-cell"
                data-model="${modelEnc}" data-cap="${capEnc}" data-skill="${skillEnc}"
                style="font-size:12px;font-weight:600;color:#2d3748;cursor:pointer;padding:2px 6px;border-radius:4px;transition:background 0.15s;"
                onmouseenter="showScoreTooltip()" onmouseleave="hideTooltip()">
                ${scoreStr}
              </span>
            </td>
          </tr>
        `);
      }
    }

    htmlParts.push(`
      <tr class="llm-row custom-row" data-target="${detailId}">
        <td style="width:8%;">${index + 1}</td>
        <td style="width:65%;"><span class="hovertip">&#9658;</span> ${formatModelName(row.name)}</td>
        <td style="width:27%;text-align:right;">${row.score.toFixed(3)}</td>
      </tr>
      <tr id="${detailId}" class="llm-details" style="display:none;">
        <td colspan="3" style="padding:0;">
          <table class="table table-sm" style="width:100%;margin:0;border-collapse:separate;border-spacing:0;">
            <colgroup>
              <col style="width:8%;"><col style="width:48%;"><col style="width:26%;"><col style="width:18%;">
            </colgroup>
            <tbody>${detailParts.join('')}</tbody>
          </table>
        </td>
      </tr>
    `);
  });

  tableBody.innerHTML = htmlParts.join('');
  attachExpandCollapse();
  attachScoreClickHandlers();
}

function attachExpandCollapse() {
  document.querySelectorAll('.llm-row').forEach(row => {
    row.addEventListener('click', function () {
      const arrow     = this.querySelector('.hovertip');
      const detailRow = document.getElementById(this.getAttribute('data-target'));
      const expanded  = detailRow.style.display === 'table-row';
      detailRow.style.display = expanded ? 'none' : 'table-row';
      arrow.innerHTML = expanded ? '&#9654;' : '&#9660;';
      arrow.style.color = '#2f855a'; arrow.style.fontWeight = 'bold';
    });
  });
}

function attachScoreClickHandlers() {
  document.querySelectorAll('.skill-score-cell').forEach(cell => {
    cell.addEventListener('click', e => {
      e.stopPropagation();
      hideTooltip();
      openSkillModal(
        decodeURIComponent(cell.dataset.model),
        decodeURIComponent(cell.dataset.cap),
        decodeURIComponent(cell.dataset.skill)
      );
    });
    cell.addEventListener('mouseenter', () => {
      cell.style.background = '#ebf8ff';
    });
    cell.addEventListener('mouseleave', () => {
      cell.style.background = '';
      hideTooltip();
    });
  });
}
