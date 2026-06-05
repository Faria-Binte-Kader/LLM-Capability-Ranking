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

// Preferred capability order (from skill_definitions.csv); any extra caps come after
const CAP_ORDER_PREFERRED = [
  'Reasoning',
  'Narrative Understanding',
  'Social-Aware Communication',
  'Creativity',
  'Ethical / Fair Response Generation',
  'Adherence to Instructions',
  'Multilingual Generation',
  'Personalization & Adaptation',
  'Decision Support',
  'Automation & Planning',
  'Data Understanding',
  'Interaction & Assistance',
  'General Cognitive Skills',
];

// ─── Display name helpers ─────────────────────────────────────────────────────
const ORG_NAMES = {
  'qwen':        'Qwen',
  'allenai':     'AllenAI',
  'deepcogito':  'DeepCogito',
  'deepseek-ai': 'DeepSeek',
  'google':      'Google',
  'meta-llama':  'Meta',
  'microsoft':   'Microsoft',
  'mistralai':   'Mistral AI',
  'tiiuae':      'TII UAE',
};

const DATASET_DISPLAY = {
  'boolq':        'BoolQ',
  'cnndm':        'CNN/DM',
  'xsum':         'XSum',
  'gsm8k':        'GSM8K',
  'humaneval':    'HumanEval',
  'imdb':         'IMDb',
  'med_qa':       'Med QA',
  'narrative_qa': 'Narrative QA',
  'truthful_qa':  'TruthfulQA',
  'legal_support':'Legal Support',
  'civil_comments':'Civil Comments',
};

const PREFIX_DISPLAY = {
  'bbq':            'BBQ',
  'mmlu':           'MMLU',
  'em':             'Entity Matching',
  'disinformation': 'Disinformation',
};

const METRIC_DISPLAY = {
  'pass@1':              'Pass@1',
  'gsm8k_exact':         'GSM8K Exact',
  'self_bleu (offline)': 'Self-BLEU (Offline)',
};

function titleCase(str) {
  return str.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatSubPart(sub) {
  const hasMixedCase = /[A-Z]/.test(sub) && /[A-Z].*[a-z]/.test(sub);
  return hasMixedCase ? sub.replace(/_/g, ' ') : titleCase(sub);
}

function formatDatasetName(raw) {
  if (DATASET_DISPLAY[raw]) return DATASET_DISPLAY[raw];
  if (raw.includes(':')) {
    const colon  = raw.indexOf(':');
    const prefix = raw.slice(0, colon);
    const sub    = raw.slice(colon + 1);
    return `${PREFIX_DISPLAY[prefix] || titleCase(prefix)}: ${formatSubPart(sub)}`;
  }
  return titleCase(raw);
}

function formatGroupName(prefix) {
  return PREFIX_DISPLAY[prefix] || titleCase(prefix);
}

function formatMetricName(raw) {
  return METRIC_DISPLAY[raw] || titleCase(raw);
}

function formatModelName(rawId) {
  const idx    = rawId.indexOf('_');
  const orgKey = (idx === -1 ? rawId : rawId.slice(0, idx)).toLowerCase();
  const model  = idx === -1 ? rawId : rawId.slice(idx + 1);
  const company = ORG_NAMES[orgKey] || (orgKey.charAt(0).toUpperCase() + orgKey.slice(1));

  let m = model
    .replace(/-instruct(?=-\d{4}$)/i, ' Instruct')
    .replace(/-(\d{4})$/, ' $1')
    .replace(/-instruct$/i, ' Instruct')
    .replace(/-preview$/i, ' Preview')
    .replace(/-it$/i, ' IT')
    .replace(/(\d+)b(?=[-\s]|$)/gi, (_, n) => n + 'B');

  m = m.charAt(0).toUpperCase() + m.slice(1);
  return `${m} (${company})`;
}

// ─── CSV parser ───────────────────────────────────────────────────────────────
// Handles quoted fields, embedded commas, and "" escaped quotes.
function parseCSV(text) {
  const rows = [];
  let i = 0;
  const n = text.length;

  while (i < n) {
    // Skip blank lines
    while (i < n && (text[i] === '\r' || text[i] === '\n')) i++;
    if (i >= n) break;

    const row = [];
    while (i < n && text[i] !== '\n' && text[i] !== '\r') {
      let field = '';
      if (text[i] === '"') {
        i++; // opening quote
        while (i < n) {
          if (text[i] === '"') {
            if (i + 1 < n && text[i + 1] === '"') { field += '"'; i += 2; }
            else { i++; break; } // closing quote
          } else {
            field += text[i++];
          }
        }
      } else {
        while (i < n && text[i] !== ',' && text[i] !== '\n' && text[i] !== '\r')
          field += text[i++];
      }
      row.push(field.trim());
      if (i < n && text[i] === ',') i++;
    }
    // skip line ending
    while (i < n && (text[i] === '\n' || text[i] === '\r')) i++;

    if (row.some(f => f !== '')) rows.push(row);
  }
  return rows;
}

// Parse CSV text into array of objects keyed by header row
function parseCSVWithHeaders(text) {
  const rows = parseCSV(text);
  if (rows.length === 0) return [];
  const headers = rows[0];
  return rows.slice(1).map(row => {
    const obj = {};
    headers.forEach((h, j) => { obj[h] = row[j] || ''; });
    return obj;
  });
}

// ─── Dataset name normalization (CSV name → results.json key(s)) ───────────────
// Known display-name overrides for dataset names that don't match results.json keys
const DISPLAY_NAME_MAP = {
  'BoolQ':       'boolq',
  'NarrativeQA': 'narrative_qa',
  'TruthfulQA':  'truthful_qa',
  'CNN/DailyMail': 'cnndm',
  'XSUM':        'xsum',
  'IMDB':        'imdb',
  'CivilComments': 'civil_comments',
  'GSM8K':       'gsm8k',
  'HumanEval':   'humaneval',
  'LegalSupport': 'legal_support',
  'MedQA':       'med_qa',
};

// Group names that expand to all results.json keys sharing a prefix
const GROUP_PREFIX_MAP = {
  'BBQ':           'bbq:',
  'EntityMatching': 'em:',
  'Disinformation (climate & covid)': 'disinformation:',
};

function resolveDatasetName(csvName, allResultKeys) {
  // 1. Direct match (e.g. "bbq:age", "em:Abt_Buy" — added verbatim in CSV)
  if (allResultKeys.has(csvName)) return [csvName];

  // 2. Known display-name substitution
  const mapped = DISPLAY_NAME_MAP[csvName];
  if (mapped) return allResultKeys.has(mapped) ? [mapped] : [];

  // 3. Group expansion (e.g. "BBQ" → all "bbq:*" keys)
  const pfx = GROUP_PREFIX_MAP[csvName];
  if (pfx) return Array.from(allResultKeys).filter(k => k.startsWith(pfx));

  // 4. MMLU subtask: bare name → "mmlu:<name>"
  const mmluKey = 'mmlu:' + csvName;
  if (allResultKeys.has(mmluKey)) return [mmluKey];

  // 5. EM subdataset: bare name → "em:<name>"
  const emKey = 'em:' + csvName;
  if (allResultKeys.has(emKey)) return [emKey];

  // 6. Lowercase + underscore normalisation fallback
  const norm = csvName.toLowerCase().replace(/[\s-]+/g, '_');
  if (allResultKeys.has(norm)) return [norm];

  return []; // unrecognised — skip
}

// ─── State ────────────────────────────────────────────────────────────────────
let llmData       = {};
let capMap        = {};   // capability → skill → [dataset_key, …] (sorted, deduped)
let skillDefs     = {};   // skill name → { def, example }
let validDatasets = new Set();
let capOrderFinal = [];   // effective capability order after data load
let selectedTemperature = '0.2';
let rankUpdateTimer     = null;

// ─── Init: load all three sources in parallel ─────────────────────────────────
Promise.all([
  fetch('results.json').then(r => r.json()),
  fetch('dataset_skill_mapping_updated.csv').then(r => r.text()),
  fetch('skill_definitions.csv').then(r => r.text()),
]).then(([results, mappingCSV, defsCSV]) => {
  llmData = results;
  buildValidDatasets();
  buildSkillDefinitions(defsCSV);
  buildCapabilityMap(mappingCSV);
  initTooltip();
  buildCapabilityPanel();
  initTemperatureSelector();
  initCheckboxHandlers();
  initGlobalToggleButtons();
  updateRankingTable();
});

// ─── Data preparation ─────────────────────────────────────────────────────────

function buildValidDatasets() {
  const models = Object.keys(llmData);
  const allDs  = new Set();
  for (const m of models)
    for (const t of Object.values(llmData[m]))
      for (const d of Object.keys(t)) allDs.add(d);

  for (const ds of allDs) {
    let count = 0;
    for (const m of models)
      for (const t of Object.values(llmData[m]))
        if (t[ds]) { count++; break; }
    if (count === models.length) validDatasets.add(ds);
  }
}

function buildSkillDefinitions(csvText) {
  skillDefs = {};
  const rows = parseCSVWithHeaders(csvText);
  for (const row of rows) {
    const skill = (row['Skill / Certificate'] || '').trim();
    const def   = (row['Definition'] || '').trim();
    const ex    = (row['Example Task / Test Case'] || '').trim();
    if (skill && def) skillDefs[skill] = { def, example: ex };
  }
}

function buildCapabilityMap(csvText) {
  capMap = {};
  const rows = parseCSVWithHeaders(csvText);

  for (const row of rows) {
    const csvName = (row['Dataset'] || '').trim();
    const cap     = (row['Capability'] || '').trim();
    const skill   = (row['Skill'] || row['Skill / Certificate'] || '').trim();

    // Skip header-like or placeholder rows (MMLU Subtask, empty cap/skill)
    if (!csvName || !cap || !skill) continue;

    const keys = resolveDatasetName(csvName, validDatasets);
    if (keys.length === 0) continue;

    if (!capMap[cap]) capMap[cap] = {};
    if (!capMap[cap][skill]) capMap[cap][skill] = new Set();
    for (const k of keys) capMap[cap][skill].add(k);
  }

  // Convert Sets to sorted arrays; drop empty skills
  for (const cap of Object.keys(capMap)) {
    for (const skill of Object.keys(capMap[cap])) {
      const arr = Array.from(capMap[cap][skill]).sort();
      if (arr.length > 0) capMap[cap][skill] = arr;
      else delete capMap[cap][skill];
    }
    if (Object.keys(capMap[cap]).length === 0) delete capMap[cap];
  }

  // Build final capability order: preferred order first, then any extras
  const seen = new Set();
  capOrderFinal = [];
  for (const c of CAP_ORDER_PREFERRED) if (capMap[c]) { capOrderFinal.push(c); seen.add(c); }
  for (const c of Object.keys(capMap)) if (!seen.has(c)) capOrderFinal.push(c);
}

// ─── Tooltip ──────────────────────────────────────────────────────────────────

function initTooltip() {
  const tt = document.createElement('div');
  tt.id = 'skill-tooltip';
  tt.style.cssText = [
    'position:fixed',
    'background:#2d3748',
    'color:#fff',
    'padding:14px 18px',
    'border-radius:10px',
    'font-size:12px',
    'max-width:340px',
    'z-index:9999',
    'display:none',
    'pointer-events:none',
    'line-height:1.6',
    'box-shadow:0 6px 24px rgba(0,0,0,0.35)',
  ].join(';');
  document.body.appendChild(tt);

  document.addEventListener('mousemove', e => {
    if (tt.style.display === 'none') return;
    const x = e.clientX + 18;
    const y = e.clientY + 12;
    tt.style.left = Math.min(x, window.innerWidth  - tt.offsetWidth  - 12) + 'px';
    tt.style.top  = Math.min(y, window.innerHeight - tt.offsetHeight - 12) + 'px';
  });
}

function showSkillTooltip(skill, datasets) {
  const tt  = document.getElementById('skill-tooltip');
  const def = skillDefs[skill];

  const defHtml = def
    ? `<span style="color:#e2e8f0;">${def.def}</span>
       <span style="display:block; margin-top:6px; color:#a0aec0; font-style:italic;">
         Example: ${def.example}
       </span>`
    : '';

  const dsNames = datasets.map(d => formatDatasetName(d)).join(', ');
  const dsHtml  = `<span style="display:block; margin-top:10px; color:#90cdf4;">
                     <strong style="color:#bee3f8;">Datasets:</strong> ${dsNames}
                   </span>`;

  tt.innerHTML = `<strong style="font-size:13px; display:block; margin-bottom:8px;">${skill}</strong>
                  ${defHtml}${dsHtml}`;
  tt.style.display = 'block';
}

function hideTooltip() {
  const tt = document.getElementById('skill-tooltip');
  if (tt) tt.style.display = 'none';
}

// ─── Left panel ───────────────────────────────────────────────────────────────

function buildCapabilityPanel() {
  const container = document.getElementById('task-list');
  container.innerHTML = '';

  for (const cap of capOrderFinal) {
    if (!capMap[cap] || Object.keys(capMap[cap]).length === 0) continue;
    const color  = CAP_COLORS[cap] || '#555';
    const skills = Object.keys(capMap[cap]);

    const capGroup = document.createElement('div');
    capGroup.className = 'task-group';
    capGroup.style.marginBottom = '10px';

    // Capability header
    const header = document.createElement('div');
    header.className = 'task-header';
    header.style.cssText = 'display:flex; align-items:center; gap:8px; padding:4px 0;';

    const capCb = document.createElement('input');
    capCb.type = 'checkbox';
    capCb.className = 'cap-cb';
    capCb.dataset.cap = cap;
    capCb.checked = true;
    capCb.style.cssText = 'cursor:pointer; flex-shrink:0;';

    const arrow = document.createElement('span');
    arrow.className = 'cap-arrow';
    arrow.innerHTML = '&#9658;';
    arrow.style.cssText = `color:${color}; font-weight:bold; width:16px; display:inline-block; flex-shrink:0; cursor:pointer;`;

    const capLabel = document.createElement('span');
    capLabel.style.cssText = `font-weight:bold; font-size:14px; flex:1; cursor:pointer; color:${color};`;
    capLabel.textContent = cap;

    header.appendChild(capCb);
    header.appendChild(arrow);
    header.appendChild(capLabel);

    // Skill list (collapsed by default)
    const skillList = document.createElement('div');
    skillList.className = 'cap-skills';
    skillList.style.cssText = 'display:none; padding-left:24px; margin-top:4px;';

    for (const skill of skills) {
      const datasets = capMap[cap][skill];

      // Skill checkbox row
      const row = document.createElement('div');
      row.style.cssText = 'display:flex; align-items:center; padding:3px 4px 1px 4px; gap:6px; cursor:default;';

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.className = 'skill-cb';
      cb.dataset.cap = cap;
      cb.dataset.skill = skill;
      cb.checked = true;
      cb.style.cssText = 'cursor:pointer; flex-shrink:0;';

      const lbl = document.createElement('label');
      lbl.style.cssText = 'font-size:13px; cursor:pointer; margin:0; flex:1;';
      lbl.innerHTML = `${skill} <small style="color:#a0aec0;">(${datasets.length} dataset${datasets.length !== 1 ? 's' : ''})</small>`;

      row.appendChild(cb);
      row.appendChild(lbl);

      // Tooltip shows skill definition + all individual dataset names
      row.addEventListener('mouseenter', () => showSkillTooltip(skill, datasets));
      row.addEventListener('mouseleave', hideTooltip);

      skillList.appendChild(row);
    }

    capGroup.appendChild(header);
    capGroup.appendChild(skillList);
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
      capCb.indeterminate = false;
      scheduleRankingUpdate();
    });
  }
}

function updateCapCheckboxState(cap) {
  const capCb    = Array.from(document.querySelectorAll('.cap-cb')).find(c => c.dataset.cap === cap);
  if (!capCb) return;
  const skillCbs = Array.from(document.querySelectorAll('.skill-cb')).filter(c => c.dataset.cap === cap);
  const checked  = skillCbs.filter(c => c.checked).length;
  capCb.indeterminate = checked > 0 && checked < skillCbs.length;
  capCb.checked       = checked > 0;
}

// ─── Initialisation helpers ───────────────────────────────────────────────────

function initTemperatureSelector() {
  const tempSelect = document.getElementById('temp');
  if (!tempSelect) return;
  selectedTemperature = tempSelect.value;
  tempSelect.addEventListener('change', () => {
    selectedTemperature = tempSelect.value;
    scheduleRankingUpdate();
  });
}

function initCheckboxHandlers() {
  document.getElementById('task-list').addEventListener('change', e => {
    if (e.target.classList.contains('skill-cb')) {
      updateCapCheckboxState(e.target.dataset.cap);
      scheduleRankingUpdate();
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

function getDatasetScore(modelData, dataset, temp) {
  for (const taskData of Object.values(modelData)) {
    if (taskData[dataset]) {
      const scores = Object.values(taskData[dataset])
        .map(t => t[temp])
        .filter(v => v !== undefined && v !== null);
      if (scores.length > 0) return scores.reduce((a, b) => a + b, 0) / scores.length;
    }
  }
  return null;
}

function getDatasetMetrics(modelData, dataset) {
  for (const taskData of Object.values(modelData))
    if (taskData[dataset]) return taskData[dataset];
  return {};
}

// Union of dataset keys across all checked skills (deduplicated)
function getSelectedDatasets() {
  const seen = new Set();
  document.querySelectorAll('.skill-cb:checked').forEach(cb => {
    for (const ds of (capMap[cb.dataset.cap]?.[cb.dataset.skill] || []))
      seen.add(ds);
  });
  return Array.from(seen);
}

// [{cap, skill, datasets}] for checked skills in capOrderFinal
function getSelectedCapSkillDatasets() {
  const checkedKeys = new Set();
  document.querySelectorAll('.skill-cb:checked').forEach(cb => {
    checkedKeys.add(`${cb.dataset.cap}||${cb.dataset.skill}`);
  });
  const result = [];
  for (const cap of capOrderFinal) {
    if (!capMap[cap]) continue;
    for (const skill of Object.keys(capMap[cap]))
      if (checkedKeys.has(`${cap}||${skill}`))
        result.push({ cap, skill, datasets: capMap[cap][skill] });
  }
  return result;
}

// Groups dataset keys by ':' prefix for collapsible rows in model detail view
function groupByPrefixKeys(datasets) {
  const prefixCount = {};
  for (const ds of datasets)
    if (ds.includes(':')) { const p = ds.split(':')[0]; prefixCount[p] = (prefixCount[p] || 0) + 1; }

  const result = [];
  const seen   = new Set();
  for (const ds of datasets) {
    if (ds.includes(':')) {
      const p = ds.split(':')[0];
      if (prefixCount[p] > 1) {
        if (!seen.has(p)) {
          seen.add(p);
          result.push({ type: 'group', prefix: p, datasets: datasets.filter(d => d.startsWith(p + ':')) });
        }
      } else if (!seen.has(ds)) {
        seen.add(ds);
        result.push({ type: 'standalone', dataset: ds });
      }
    } else if (!seen.has(ds)) {
      seen.add(ds);
      result.push({ type: 'standalone', dataset: ds });
    }
  }
  return result;
}

// ─── Ranking table ────────────────────────────────────────────────────────────

function scheduleRankingUpdate() {
  clearTimeout(rankUpdateTimer);
  rankUpdateTimer = setTimeout(updateRankingTable, 180);
}

function updateRankingTable() {
  const selected = getSelectedDatasets();
  if (selected.length === 0) { showEmptyMessage(); return; }
  renderRankingTable(selected);
}

function showEmptyMessage() {
  document.querySelector('#llm-table-body').innerHTML = `
    <tr>
      <td colspan="3" style="text-align:center; padding:20px; font-style:italic;">
        Please select at least one skill to rank models.
      </td>
    </tr>
  `;
}

function renderRankingTable(selectedDatasets) {
  const tableBody         = document.querySelector('#llm-table-body');
  const selectedCapSkills = getSelectedCapSkillDatasets();

  // Overall score = aggregate (sum) of unique selected dataset scores per model
  const rows = [];
  for (const [modelName, modelData] of Object.entries(llmData)) {
    let total = 0, count = 0;
    for (const ds of selectedDatasets) {
      const s = getDatasetScore(modelData, ds, selectedTemperature);
      if (s !== null) { total += s; count++; }
    }
    if (count === 0) continue;
    rows.push({ name: modelName, score: total });
  }
  rows.sort((a, b) => b.score - a.score);

  let gidCounter = 0;
  const htmlParts = [];

  rows.forEach((row, index) => {
    const detailId  = `llm-detail-${index + 1}`;
    const modelData = llmData[row.name];

    // Group selected cap-skills by capability for the detail view
    const byCap    = {};
    const capOrder = [];
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
          <td colspan="3" style="font-size:12px; font-weight:bold; color:${capColor};
              padding-top:10px; padding-bottom:2px; border-top:1px solid #e2e8f0;">
            ${cap}
          </td>
        </tr>
      `);

      for (const { skill, datasets } of byCap[cap]) {
        let skillSum = 0, skillCount = 0;
        for (const ds of datasets) {
          const s = getDatasetScore(modelData, ds, selectedTemperature);
          if (s !== null) { skillSum += s; skillCount++; }
        }

        detailParts.push(`
          <tr style="background-color:#f7fafc;">
            <td></td>
            <td colspan="2" style="font-size:12px; font-weight:600; color:#2d3748;
                padding-left:12px; padding-top:6px; padding-bottom:4px;">
              ${skill}
            </td>
            <td style="text-align:right; font-size:12px; font-weight:600; color:#2d3748;">
              ${skillCount > 0 ? skillSum.toFixed(3) : 'N/A'}
            </td>
          </tr>
        `);

        for (const item of groupByPrefixKeys(datasets)) {
          if (item.type === 'standalone') {
            const metrics = getDatasetMetrics(modelData, item.dataset);
            for (const [metric, temps] of Object.entries(metrics)) {
              const val = temps[selectedTemperature];
              detailParts.push(`
                <tr>
                  <td></td>
                  <td style="font-size:12px; padding-left:24px;">${formatDatasetName(item.dataset)}</td>
                  <td style="font-size:12px; color:#718096;">${formatMetricName(metric)}</td>
                  <td style="text-align:right; font-size:12px;">${val !== undefined ? Number(val).toFixed(3) : 'N/A'}</td>
                </tr>
              `);
            }
          } else {
            const gid = `gid-${++gidCounter}`;
            let groupSum = 0, groupCount = 0;
            for (const ds of item.datasets) {
              const s = getDatasetScore(modelData, ds, selectedTemperature);
              if (s !== null) { groupSum += s; groupCount++; }
            }
            detailParts.push(`
              <tr class="dataset-group-row" data-gid="${gid}" style="cursor:pointer; background-color:#f7fafc;">
                <td></td>
                <td style="font-size:12px; font-weight:600; padding-left:24px;">
                  <span class="group-arrow" style="color:#2f855a; font-weight:bold; width:16px; display:inline-block;">&#9658;</span>
                  ${formatGroupName(item.prefix)}
                  <small style="color:#a0aec0; font-weight:400;">(${item.datasets.length})</small>
                </td>
                <td style="font-size:11px; color:#a0aec0;">sum</td>
                <td style="text-align:right; font-size:12px; font-weight:600;">
                  ${groupCount > 0 ? groupSum.toFixed(3) : 'N/A'}
                </td>
              </tr>
            `);
            for (const ds of item.datasets) {
              const metrics = getDatasetMetrics(modelData, ds);
              for (const [metric, temps] of Object.entries(metrics)) {
                const val = temps[selectedTemperature];
                detailParts.push(`
                  <tr class="group-sub-row" data-gid="${gid}" style="display:none; background-color:#eaffea;">
                    <td></td>
                    <td style="font-size:12px; padding-left:44px;">${formatDatasetName(ds)}</td>
                    <td style="font-size:12px; color:#718096;">${formatMetricName(metric)}</td>
                    <td style="text-align:right; font-size:12px;">${val !== undefined ? Number(val).toFixed(3) : 'N/A'}</td>
                  </tr>
                `);
              }
            }
          }
        }
      }
    }

    htmlParts.push(`
      <tr class="llm-row custom-row" data-target="${detailId}">
        <td style="width:8%;">${index + 1}</td>
        <td style="width:65%;"><span class="hovertip">&#9658;</span> ${formatModelName(row.name)}</td>
        <td style="width:27%; text-align:right;">${row.score.toFixed(3)}</td>
      </tr>
      <tr id="${detailId}" class="llm-details" style="display:none;">
        <td colspan="3" style="padding:0;">
          <table class="table table-sm" style="width:100%; margin:0; border-collapse:separate; border-spacing:0;">
            <colgroup>
              <col style="width:8%;">
              <col style="width:43%;">
              <col style="width:27%;">
              <col style="width:22%;">
            </colgroup>
            <thead>
              <tr>
                <td></td>
                <th>Dataset</th>
                <th>Metric</th>
                <th style="text-align:right;">Score</th>
              </tr>
            </thead>
            <tbody>${detailParts.join('')}</tbody>
          </table>
        </td>
      </tr>
    `);
  });

  tableBody.innerHTML = htmlParts.join('');
  attachExpandCollapse();
  attachGroupExpandCollapse();
}

// ─── Event attachment ─────────────────────────────────────────────────────────

function attachExpandCollapse() {
  document.querySelectorAll('.llm-row').forEach(row => {
    row.addEventListener('click', function () {
      const arrow     = this.querySelector('.hovertip');
      const detailRow = document.getElementById(this.getAttribute('data-target'));
      const expanded  = detailRow.style.display === 'table-row';
      detailRow.style.display = expanded ? 'none' : 'table-row';
      arrow.innerHTML         = expanded ? '&#9654;' : '&#9660;';
      arrow.style.color       = '#2f855a';
      arrow.style.fontWeight  = 'bold';
    });
  });
}

function attachGroupExpandCollapse() {
  document.querySelectorAll('.dataset-group-row').forEach(groupRow => {
    groupRow.addEventListener('click', function () {
      const gid     = this.dataset.gid;
      const arrow   = this.querySelector('.group-arrow');
      const subRows = document.querySelectorAll(`.group-sub-row[data-gid="${gid}"]`);
      const expanded = subRows.length > 0 && subRows[0].style.display !== 'none';
      subRows.forEach(r => (r.style.display = expanded ? 'none' : 'table-row'));
      arrow.innerHTML = expanded ? '&#9658;' : '&#9660;';
    });
  });
}
