import { TASKS } from './examples.js';
import { $, el, icon, iconButton } from './dom.js';
import { createToolsEditor } from './tools-editor.js';

const MODEL_URLS = ['model.cact', '../models/model.cact']; // deployed copy, then the repo copy for local use
const CACHE_NAME = 'spindle-model-v1';

// ---------------------------------------------------------------- model loading
const worker = new Worker('worker.js', { type: 'module' });
let ready = false, pending = null, nextId = 1;

function setStatus(text, { progress = null, state = '' } = {}) {
  $('status-text').textContent = text;
  $('status').className = 'status ' + state;
  $('status').title = state === 'failed' ? text : 'Runs entirely in your browser. Nothing you type leaves this page.';
  $('status-bar').hidden = progress === null;
  if (progress !== null) $('status-fill').style.transform = `scaleX(${progress})`;
}

async function fetchModel() {
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch { /* private mode: no cache */ }
  for (const url of MODEL_URLS) {
    const cached = cache && await cache.match(url);
    if (cached) { setStatus('Loading model…'); return cached.arrayBuffer(); }
  }
  let lastError = null;
  for (const url of MODEL_URLS) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
      const total = Number(response.headers.get('content-length')) || 0;
      const chunks = []; let received = 0;
      const reader = response.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); received += value.length;
        const mb = (n) => (n / 1048576).toFixed(0);
        setStatus(total ? `Downloading model ${mb(received)} / ${mb(total)} MB` : `Downloading model ${mb(received)} MB`,
          { progress: total ? received / total : null });
      }
      const bytes = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
      if (cache) cache.put(url, new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } })).catch(() => {});
      return bytes.buffer;
    } catch (error) { lastError = error; }
  }
  throw lastError;
}

worker.onmessage = ({ data }) => {
  if (data.type === 'ready') {
    ready = true;
    setStatus('Ready', { state: 'ready' });
    updateRun();
    warm();
  } else if (pending && data.id === pending.id) {
    if (data.type === 'progress') $('progress').textContent = `${data.tokens} tokens`;
    else if (data.type === 'result') pending.resolve(data.result);
    else if (data.type === 'error') pending.reject(new Error(data.message));
  } else if (data.type === 'error') {
    setStatus('Model failed to start', { state: 'failed' });
  }
};
worker.onerror = (event) => setStatus('Engine failed to start', { state: 'failed' });

fetchModel().then((buffer) => {
  setStatus('Starting…', { progress: 1 });
  worker.postMessage({ type: 'init', model: buffer }, [buffer]);
}).catch((error) => setStatus('Download failed', { state: 'failed' }));

// ---------------------------------------------------------------- settings
const DEFAULT_SETTINGS = { window: 0, maxNewTokens: 256, minCertainty: 0, filters: true };
let settings = { ...DEFAULT_SETTINGS };
try { Object.assign(settings, JSON.parse(localStorage.getItem('spindle-settings') ?? '{}')); } catch { /* storage unavailable */ }

// What the worker needs from the settings.
const engineSettings = () => ({
  window: settings.window, maxNewTokens: settings.maxNewTokens, filters: settings.filters, minCallProbability: settings.minCertainty / 100,
});

function showSettings() {
  $('s-window').value = settings.window;
  $('s-tokens').value = settings.maxNewTokens;
  $('s-certainty').value = settings.minCertainty;
  $('s-certainty-out').textContent = `${settings.minCertainty}%`;
  $('s-filters').checked = settings.filters;
}

function saveSettings() {
  settings = {
    window: Number($('s-window').value), maxNewTokens: Number($('s-tokens').value),
    minCertainty: Number($('s-certainty').value), filters: $('s-filters').checked,
  };
  $('s-certainty-out').textContent = `${settings.minCertainty}%`;
  try { localStorage.setItem('spindle-settings', JSON.stringify(settings)); } catch { /* storage unavailable */ }
  warm();
}

$('settings-btn').append(icon('gear'));
for (const id of ['s-window', 's-tokens', 's-certainty', 's-filters']) $(id).addEventListener('input', saveSettings);
$('settings-reset').onclick = () => { settings = { ...DEFAULT_SETTINGS }; showSettings(); saveSettings(); };

function toggleSettings(open) {
  $('settings').hidden = !open;
  $('settings-btn').setAttribute('aria-expanded', String(open));
}
$('settings-btn').onclick = () => toggleSettings($('settings').hidden);
document.addEventListener('click', (event) => { if (!event.target.closest('.menu')) toggleSettings(false); });
document.addEventListener('keydown', (event) => { if (event.key === 'Escape') toggleSettings(false); });
showSettings();

// ---------------------------------------------------------------- UI state
let task = TASKS[0], set = task.sets[0], tools = [], originalTools = '', busy = false, warmTimer = null;
const editor = createToolsEditor($('tools-visual'), { onChange: toolsChanged });
const MAX_CHIPS = 4;

function renderTabs() {
  $('tabs').replaceChildren(...TASKS.map((t) => {
    const button = el('button', '', t.title);
    button.type = 'button';
    button.setAttribute('role', 'tab');
    button.setAttribute('aria-selected', String(t === task));
    button.onclick = () => selectTask(t);
    return button;
  }));
}

function renderSets() {
  $('sets').hidden = task.sets.length < 2;
  $('sets').replaceChildren(...task.sets.map((s) => {
    const button = el('button', '', s.name);
    button.type = 'button';
    button.setAttribute('aria-pressed', String(s === set));
    button.onclick = () => selectSet(s);
    return button;
  }));
}

function renderExamples(all = false) {
  const shown = all ? set.examples : set.examples.slice(0, MAX_CHIPS);
  const chips = shown.map((example) => {
    const chip = el('button', '', example.label);
    chip.type = 'button';
    chip.onclick = () => { $('input').value = example.text; $('input').focus(); };
    return chip;
  });
  if (!all && set.examples.length > MAX_CHIPS) {
    const more = el('button', 'more', `+${set.examples.length - MAX_CHIPS} more`);
    more.type = 'button';
    more.onclick = () => renderExamples(true);
    chips.push(more);
  }
  $('examples').replaceChildren(...chips);
}

function selectTask(t) {
  task = t;
  renderTabs();
  $('tools-title').textContent = t.mode === 'calls' ? 'Tools' : 'Schema';
  $('tools-add').textContent = t.mode === 'calls' ? '+ Add tool' : '+ Add schema';
  $('input').placeholder = t.mode === 'calls' ? 'Describe what you want done…' : 'Paste some text…';
  $('input').setAttribute('aria-label', t.mode === 'calls' ? 'Request' : 'Text');
  selectSet(t.sets[0]);
}

function selectSet(s) {
  set = s;
  renderSets();
  renderExamples();
  tools = structuredClone(s.tools);
  originalTools = JSON.stringify(tools);
  editor.set(tools);
  showTools();
  $('input').value = s.examples[0].text;
  $('result').hidden = true;
  warm();
}

$('input').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) { event.preventDefault(); run(); }
});
$('run').onclick = run;
$('run-key').textContent = /Mac|iPhone|iPad/.test(navigator.platform) ? '⌘↵' : 'Ctrl ↵';

function updateRun() {
  $('run').disabled = !ready || busy || Boolean(toolsProblem());
  $('run').classList.toggle('busy', busy);
  $('run-label').textContent = busy ? 'Running…' : 'Run';
}

// ---------------------------------------------------------------- tools panel
let mode = 'visual';

function toolsProblem() {
  if (!tools.length) return 'Add at least one tool.';
  const names = tools.map((t) => t.name);
  if (names.some((name) => !name)) return 'Every tool needs a name.';
  if (new Set(names).size < names.length) return 'Tool names must be different.';
  return '';
}

function showTools(problem = toolsProblem()) {
  $('tools-count').textContent = String(tools.length);
  $('tools-reset').hidden = JSON.stringify(tools) === originalTools;
  $('tools-error').hidden = !problem;
  $('tools-error').textContent = problem;
  updateRun();
}

function toolsChanged() {
  $('tools-json').value = JSON.stringify(tools, null, 2);
  showTools();
  clearTimeout(warmTimer);
  warmTimer = setTimeout(warm, 600);
}

function setMode(next) {
  if (next === 'visual' && mode === 'json' && !readJson()) return;
  mode = next;
  for (const button of $('tools-mode').children) button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
  $('tools-visual').hidden = $('tools-add').hidden = mode !== 'visual';
  $('tools-json').hidden = mode !== 'json';
  $('tools-json').value = JSON.stringify(tools, null, 2);
}

// Read the JSON box into `tools`. Returns false (and says why) when it is not a valid list.
function readJson() {
  try {
    const parsed = JSON.parse($('tools-json').value);
    if (!Array.isArray(parsed) || !parsed.every((t) => t && typeof t === 'object')) throw new Error('Expected a list of tools.');
    tools = parsed;
    editor.set(tools);
    showTools();
    return true;
  } catch (error) {
    showTools(`Not valid JSON yet: ${error.message}`);
    return false;
  }
}

$('tools-mode').onclick = (event) => { const mode = event.target.dataset?.mode; if (mode) setMode(mode); };
$('tools-json').addEventListener('input', () => { if (readJson()) { clearTimeout(warmTimer); warmTimer = setTimeout(warm, 600); } });
$('tools-add').onclick = () => editor.add();
$('tools-reset').onclick = () => { tools = JSON.parse(originalTools); editor.set(tools); toolsChanged(); };

// ---------------------------------------------------------------- running
function warm() {
  if (!ready || busy || toolsProblem()) return;
  worker.postMessage({ type: 'warm', tools, system: '', settings: engineSettings() });
}

async function run() {
  const query = $('input').value.trim();
  if (!query || !ready || busy || toolsProblem()) return;
  busy = true; updateRun();
  $('progress').textContent = 'Thinking…';
  const id = nextId++;
  try {
    const result = await new Promise((resolve, reject) => {
      pending = { id, resolve, reject };
      worker.postMessage({ type: 'run', id, tools, system: '', query, settings: engineSettings() });
    });
    renderResult(result, tools);
  } catch (error) {
    $('result').hidden = false;
    $('result').replaceChildren(el('p', 'note error', error.message));
  } finally {
    pending = null; busy = false; updateRun(); $('progress').textContent = '';
  }
}

// ---------------------------------------------------------------- results
function valueNode(value) {
  if (typeof value === 'boolean') return el('span', 'flag', String(value));
  if (typeof value === 'number') return el('span', 'num', String(value));
  if (typeof value === 'string') return el('span', '', value);
  return el('code', 'num', JSON.stringify(value));
}

function typeTag(schema) {
  if (!schema) return null;
  const label = schema.enum?.length ? 'choice' : schema.type;
  return label ? el('span', `ptype t-${schema.type}`, label) : null;
}

// Rows for every argument. With showMissing, schema fields the model left empty are listed too.
function argumentList(call, schema, showMissing = false) {
  const list = el('dl', 'args');
  const properties = schema?.properties ?? {};
  const keys = showMissing ? [...Object.keys(properties), ...Object.keys(call.arguments).filter((k) => !(k in properties))] : Object.keys(call.arguments);
  for (const key of keys) {
    const dt = el('dt', '', key), dd = el('dd');
    const tag = typeTag(properties[key]);
    if (tag) dt.append(tag);
    if (key in call.arguments) dd.append(valueNode(call.arguments[key]));
    else { dd.className = 'missing'; dd.textContent = '—'; dd.title = 'not found in the text'; }
    list.append(dt, dd);
  }
  return list;
}

function callCard(call, index, { schema, count, held = null }) {
  const card = el('article', held ? 'call held' : 'call');
  card.style.setProperty('--i', index);
  const head = el('div', 'call-head');
  if (count > 1) head.append(el('span', 'step', String(index + 1)));
  if (task.mode === 'calls' || !call.name.startsWith('classify')) head.append(el('code', '', call.name));
  if (held) head.append(el('span', 'muted small', 'held back'));
  if (head.childNodes.length) card.append(head);

  if (call.name.startsWith('classify') && task.mode !== 'calls') {
    const list = el('dl', 'args');
    for (const [key, value] of Object.entries(call.arguments)) {
      const dd = el('dd'); dd.append(el('span', 'badge', String(value)));
      list.append(el('dt', '', key), dd);
    }
    card.append(list);
  } else if (Object.keys(call.arguments).length || task.mode !== 'calls') {
    card.append(argumentList(call, task.mode === 'calls' ? null : schema, task.mode !== 'calls'));
  } else card.append(el('p', 'muted small', 'No arguments'));
  if (held) card.append(el('p', 'why', held));
  return card;
}

function renderResult(result, tools) {
  const box = $('result');
  box.hidden = false;
  const schemaOf = (name) => tools.find((t) => t.name === name)?.parameters;

  const m = result.metrics;
  const meta = el('div', 'meta');
  const facts = [`${m.totalSeconds.toFixed(1)} s`, `${m.generatedTokens} tokens`];
  if (result.callProbabilities.length) facts.push(`${Math.round(Math.min(...result.callProbabilities) * 100)}% certain`);
  if (!result.complete) facts.push('cut off');
  for (const fact of facts) meta.append(el('span', '', fact));

  const head = el('div', 'result-head');
  const view = el('div', 'seg');
  view.setAttribute('role', 'group');
  const pretty = el('button', '', 'Calls'), raw = el('button', '', 'JSON');
  for (const b of [pretty, raw]) b.type = 'button';
  view.append(pretty, raw);
  head.append(el('h2', '', 'Result'), meta, view);

  const calls = el('div', 'calls');
  if (!result.functionCalls.length && !result.suppressedCalls.length) {
    calls.append(el('p', 'note', task.mode === 'calls' ? 'No tool fits this request, so the model made no call.' : 'No matching information found.'));
  }
  result.functionCalls.forEach((call, i) => calls.append(callCard(call, i, { schema: schemaOf(call.name), count: result.functionCalls.length })));
  result.suppressedCalls.forEach((call, i) => calls.append(callCard(call, result.functionCalls.length + i,
    { schema: schemaOf(call.name), count: 1, held: result.suppressedReasons[i] })));

  const json = JSON.stringify(result.functionCalls, null, 2);
  const rawBox = el('div', 'rawwrap');
  rawBox.hidden = true;
  const copy = iconButton('copy', 'Copy JSON', () => navigator.clipboard?.writeText(json));
  rawBox.append(el('pre', '', json), copy);

  const choose = (showRaw) => {
    pretty.setAttribute('aria-pressed', String(!showRaw)); raw.setAttribute('aria-pressed', String(showRaw));
    calls.hidden = showRaw; rawBox.hidden = !showRaw;
  };
  pretty.onclick = () => choose(false); raw.onclick = () => choose(true);
  choose(false);
  box.replaceChildren(head, calls, rawBox);
}

renderTabs();
selectTask(TASKS[0]);

// On GitHub Pages (owner.github.io/repo/) link back to the repository.
if (location.hostname.endsWith('.github.io')) {
  const repo = location.pathname.split('/')[1];
  $('source').href = `https://github.com/${location.hostname.split('.')[0]}/${repo}`;
  $('source').hidden = false;
}
