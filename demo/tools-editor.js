// Visual editor for a list of tool definitions: easy to read at a glance, editable in place.
// It changes the list it is given directly and calls onChange after every edit.
import { el, icon, iconButton } from './dom.js';

const TYPES = ['string', 'integer', 'number', 'boolean'];

const parametersOf = (tool) => (tool.parameters ??= { type: 'object', properties: {}, required: [] });
const propertiesOf = (tool) => (parametersOf(tool).properties ??= {});
const requiredOf = (tool) => (parametersOf(tool).required ??= []);
const typeLabel = (schema) => (schema.enum?.length ? schema.enum.join(' | ') : schema.type ?? 'any');

function uniqueName(base, taken) {
  let name = base, n = 2;
  while (taken.includes(name)) name = `${base}_${n++}`;
  return name;
}

export function createToolsEditor(root, { onChange }) {
  let tools = [];
  const editing = new WeakSet();

  const changed = (structural = false) => { onChange(); if (structural) render(); };

  const field = (value, placeholder, className = '') => {
    const input = el('input', className);
    input.type = 'text'; input.value = value; input.placeholder = placeholder; input.spellcheck = false;
    return input;
  };

  // ------------------------------------------------------------ read-only view
  function parameterList(tool) {
    const keys = Object.keys(propertiesOf(tool));
    if (!keys.length) return el('p', 'muted small', 'No parameters');
    const list = el('ul', 'params');
    for (const key of keys) {
      const schema = propertiesOf(tool)[key];
      const item = el('li');
      if (schema.description) item.title = schema.description;
      const name = el('span', 'pname', key);
      if (requiredOf(tool).includes(key)) name.append(el('i', 'req'));
      const type = el('span', `ptype t-${schema.type ?? 'any'}`, typeLabel(schema));
      type.title = typeLabel(schema);
      item.append(name, type);
      list.append(item);
    }
    return list;
  }

  // ------------------------------------------------------------ edit view
  function rename(tool, key, next, input) {
    const props = propertiesOf(tool);
    if (!next || next === key || next in props) { input.value = key; return; }
    parametersOf(tool).properties = Object.fromEntries(Object.entries(props).map(([k, v]) => [k === key ? next : k, v]));
    const required = requiredOf(tool), at = required.indexOf(key);
    if (at >= 0) required[at] = next;
    changed(true);
  }

  function parameterRow(tool, key) {
    const schema = propertiesOf(tool)[key];
    const row = el('div', 'prow');

    const name = field(key, 'name', 'mono');
    name.onchange = () => rename(tool, key, name.value.trim(), name);

    const type = el('select');
    const known = TYPES.includes(schema.type) ? TYPES : [...TYPES, schema.type ?? 'any'];
    for (const t of known) type.add(new Option(t, t));
    type.value = schema.type ?? 'any';
    type.onchange = () => { schema.type = type.value; if (type.value !== 'string') delete schema.enum; changed(true); };

    const required = el('label', 'check');
    const box = el('input'); box.type = 'checkbox'; box.checked = requiredOf(tool).includes(key);
    box.onchange = () => {
      const list = requiredOf(tool);
      parametersOf(tool).required = box.checked ? [...list, key] : list.filter((k) => k !== key);
      changed();
    };
    required.append(box, el('span', '', 'required'));

    const remove = iconButton('trash', `Remove ${key}`, () => {
      delete propertiesOf(tool)[key];
      parametersOf(tool).required = requiredOf(tool).filter((k) => k !== key);
      changed(true);
    }, 'danger');

    const description = field(schema.description ?? '', 'description', 'wide');
    description.oninput = () => { if (description.value) schema.description = description.value; else delete schema.description; changed(); };

    row.append(name, type, remove, description);
    if (schema.type === 'string' || schema.enum) {
      const values = field((schema.enum ?? []).join(', '), 'allowed values, comma separated (optional)', 'wide');
      values.onchange = () => {
        const list = values.value.split(',').map((v) => v.trim()).filter(Boolean);
        if (list.length) schema.enum = list; else delete schema.enum;
        changed();
      };
      row.append(values);
    }
    row.append(required);
    return row;
  }

  function parameterEditor(tool) {
    const box = el('div', 'peditor');
    for (const key of Object.keys(propertiesOf(tool))) box.append(parameterRow(tool, key));
    const add = el('button', 'ghost small');
    add.type = 'button';
    add.append(icon('plus'), document.createTextNode('Add parameter'));
    add.onclick = () => {
      const props = propertiesOf(tool);
      props[uniqueName('param', Object.keys(props))] = { type: 'string', description: '' };
      changed(true);
    };
    box.append(add);
    return box;
  }

  // ------------------------------------------------------------ one tool
  function toolCard(tool) {
    const isEditing = editing.has(tool);
    const card = el('article', `tool${isEditing ? ' editing' : ''}`);
    const head = el('div', 'tool-head');
    const titles = el('div', 'tool-titles');

    if (isEditing) {
      const name = field(tool.name ?? '', 'tool_name', 'mono strong');
      name.oninput = () => { tool.name = name.value.trim(); changed(); };
      const description = field(tool.description ?? '', 'What does this tool do?', 'wide');
      description.oninput = () => { tool.description = description.value; changed(); };
      titles.append(name, description);
    } else {
      titles.append(el('code', 'tool-name', tool.name || 'unnamed'));
      if (tool.description) titles.append(el('p', 'tool-desc', tool.description));
    }

    const actions = el('div', 'tool-actions');
    actions.append(
      isEditing
        ? iconButton('check', 'Done editing', () => { editing.delete(tool); render(); })
        : iconButton('edit', `Edit ${tool.name}`, () => { editing.add(tool); render(); }),
      iconButton('trash', `Remove ${tool.name}`, () => { tools.splice(tools.indexOf(tool), 1); changed(true); }, 'danger'),
    );
    head.append(titles, actions);
    card.append(head, isEditing ? parameterEditor(tool) : parameterList(tool));
    return card;
  }

  function render() {
    root.replaceChildren(...tools.map(toolCard));
  }

  return {
    set(list) { tools = list; render(); },
    add() {
      const tool = { name: uniqueName('new_tool', tools.map((t) => t.name)), description: '', parameters: { type: 'object', properties: {}, required: [] } };
      tools.push(tool); editing.add(tool);
      changed(true);
      root.lastElementChild?.querySelector('input')?.select();
    },
  };
}
