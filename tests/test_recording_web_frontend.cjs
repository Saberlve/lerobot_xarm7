// Behavior tests for browser shortcuts using a minimal DOM, without devices.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { randomUUID } = require('node:crypto');
const page = process.argv[2] || path.join(__dirname, '../src/lerobot_robot_ufactory/utils/webapp/recording_web_assets/index.html');
const html = fs.readFileSync(page, 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Also fail on syntax errors in the shipped asset.

class Element {
  constructor() {
    this.textContent = ''; this.innerHTML = ''; this.value = ''; this.checked = false;
    this.disabled = false; this.className = ''; this.dataset = {}; this.children = [];
    const classes = new Set();
    this.classList = {
      add(name) { classes.add(name); }, remove(name) { classes.delete(name); },
      contains(name) { return classes.has(name); },
      toggle(name, force) {
        const enabled = force === undefined ? !classes.has(name) : force;
        if (enabled) classes.add(name); else classes.delete(name);
        return enabled;
      },
    };
  }
  append(...items) { this.children.push(...items); }
  replaceChildren() { this.children = []; }
  querySelector(selector) {
    for (const child of this.children) {
      if (!(child instanceof Element)) continue;
      if (child.type === 'radio' && (selector === 'input' || selector === 'input:checked' && child.checked)) return child;
      const found = child.querySelector(selector);
      if (found) return found;
    }
    return null;
  }
  querySelectorAll() { return []; }
  closest() { return null; }
  focus() {}
  showModal() { this.open = true; }
  close() { this.open = false; }
  removeAttribute(name) { delete this[name]; }
}
const elements = new Map();
const events = {};
const windowEvents = {};
const sockets = [];
class Socket {
  static OPEN = 1;
  constructor(url) { this.url = url; this.readyState = 1; this.sent = []; sockets.push(this); }
  send(message) { this.sent.push(JSON.parse(message)); }
}
const context = vm.createContext({
  document: {
    hidden: false,
    getElementById(id) { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); },
    createElement() { return new Element(); },
    createTextNode(text) { return text; },
    addEventListener(name, listener) { events[name] = listener; },
  },
  window: { addEventListener(name, listener) { windowEvents[name] = listener; } },
  location: { protocol: 'http:', host: 'test.local' },
  WebSocket: Socket,
  fetch: async () => ({ ok: true, json: async () => ({ configs: [] }) }),
  crypto: { randomUUID }, Date, Math, JSON, Set, URL, Blob, Uint8Array, DataView, TextDecoder,
  setInterval() {}, setTimeout() {}, clearTimeout() {}, confirm: () => true,
});
vm.runInContext(script, context);
const control = sockets[0];
control.onopen();
function snapshot(phase='recording', extra={}) {
  control.onmessage({ data: JSON.stringify({ type: 'snapshot', controller: true, client_id: 'owner',
    can_claim: false, simulate: true, effective: null,
    state: { phase, session_id: 'session', version: 1, gripper_mode: 'keyboard', j7_enabled: true,
      joint_mode: 'all', frames: 1, has_unsaved: true, ...extra } }) });
}
function key(name, code, repeat=false, typing=false) {
  const event = { code, repeat, target: { closest: () => typing ? {} : null },
    prevented: false, preventDefault() { this.prevented = true; } };
  events[name](event);
  return event;
}
snapshot();
let before = control.sent.length;
assert.equal(key('keydown', 'KeyC').prevented, true);
key('keydown', 'KeyC', true);
assert.equal(control.sent.length, before + 1, 'Held key must not emit repeats');
assert.equal(control.sent.at(-1).pressed, true);
key('keyup', 'KeyC', false, true);
assert.equal(control.sent.at(-1).pressed, false, 'Release must work even after focus enters input');

before = control.sent.length;
key('keydown', 'Space', false, true);
key('keydown', 'KeyS', false, true);
assert.equal(control.sent.length, before, 'Typing in editors must not trigger robot shortcuts');

key('keydown', 'KeyS');
key('keydown', 'KeyS', true);
assert.equal(control.sent.at(-1).action, 'joint_mode');
const switches = control.sent.filter(item => item.action === 'joint_mode').length;
key('keyup', 'KeyS');
assert.equal(switches, 1, 'S hold must switch once');
snapshot('recording', { joint_mode: 'j7' });
assert.equal(elements.get('jointMode').textContent, '仅 J7');
assert.match(elements.get('jointModeButton').innerHTML, /恢复全部关节/);

before = control.sent.length;
snapshot('ready');
key('keydown', 'KeyS');
key('keydown', 'KeyC');
assert.equal(control.sent.length, before, 'Motion controls must be disabled while idle');
assert.equal(key('keydown', 'Space').prevented, true);
assert.equal(control.sent.at(-1).action, 'start');
key('keyup', 'Space');

snapshot('recording');
key('keydown', 'KeyO');
windowEvents.blur();
assert.equal(control.sent.at(-1).action, 'release', 'Blur must release gripper keys');
snapshot('recording', { gripper_mode: 'gello' });
before = control.sent.length;
key('keydown', 'KeyC');
assert.equal(control.sent.length, before, 'GELLO source must disable browser gripper');

snapshot('paused');
assert.equal(elements.get('start').disabled, true);
assert.equal(elements.get('save').disabled, false);
assert.equal(elements.get('discard').disabled, false);
snapshot('saving');
assert.equal(elements.get('save').disabled, true);
assert.equal(elements.get('discard').disabled, true);
assert.equal(elements.get('closeGripper').disabled, true);
console.log('Frontend behavior checks passed: focus isolation, deduplication, release, J7 feedback, source gating, phase gating.');

// Arrow keys send one save/discard command without an episode decision dialog.
assert.doesNotMatch(html, /episodeDialog|renderEpisodeDecision/);
snapshot('recording');
before = control.sent.length;
key('keydown', 'ArrowRight');
key('keydown', 'ArrowRight', true);
assert.equal(control.sent.length, before + 1);
assert.equal(control.sent.at(-1).action, 'save');
key('keyup', 'ArrowRight');
snapshot('recording');
before = control.sent.length;
key('keydown', 'ArrowLeft');
key('keydown', 'ArrowLeft', true);
assert.equal(control.sent.length, before + 1);
assert.equal(control.sent.at(-1).action, 'discard');
key('keyup', 'ArrowLeft');
snapshot('saving');
before = control.sent.length;
key('keydown', 'ArrowRight');
key('keydown', 'ArrowLeft');
assert.equal(control.sent.length, before, 'Save/discard keys must be disabled during saving');
console.log('Single-command arrow save/discard checks passed.');

// Offline processing must stay bound to the selected saved configuration.
assert.match(html, /id="exit"[^>]*>退出录制并启动后处理/);
assert.ok(html.indexOf('id="postprocessProgress"') < html.indexOf('id="controlBar"'),
  'Postprocessing progress must appear in the top session status panel');
snapshot('ready', {has_unsaved:false});
before = control.sent.length;
elements.get('exit').onclick();
assert.equal(control.sent.length, before + 1);
assert.equal(control.sent.at(-1).action, 'exit');
snapshot('stopping', {has_unsaved:false});
assert.equal(elements.get('postprocessProgress').classList.contains('hidden'), false);
assert.equal(elements.get('postprocessBar').value, undefined, 'Show indeterminate progress until totals arrive');
assert.match(elements.get('postprocessDetail').textContent, /释放设备.*后处理/);
snapshot('postprocessing', {has_unsaved:false});
assert.equal(elements.get('postprocessProgress').classList.contains('hidden'), false);
assert.match(elements.get('postprocessDetail').textContent, /准备后处理/);

vm.runInContext(`
  selected={path:'tasks/a.yaml',revision:'rev-a',text:'config-a'};loadedText=selected.text;
  $('yamlEditor').value=loadedText;$('configPath').value=selected.path;
  processingStatus={path:selected.path,revision:selected.revision,root:'/datasets/a',
    ready:true,processed_episodes:1,pending_episodes:2,pending_frames:60};
  state={phase:'idle'};renderPostprocessing();
`, context);
assert.equal(elements.get('postprocess').disabled, false);
assert.match(elements.get('postprocessStatus').textContent, /待处理 2 条 \/ 60 帧/);
elements.get('yamlEditor').value = 'unsaved';
vm.runInContext('updateDirty()', context);
assert.equal(elements.get('postprocess').disabled, true);
assert.match(elements.get('postprocessStatus').textContent, /保存配置/);
elements.get('yamlEditor').value = 'config-a';
vm.runInContext(`selected={path:'tasks/b.yaml',revision:'rev-b'};$('configPath').value=selected.path;renderPostprocessing()`, context);
assert.equal(elements.get('postprocess').disabled, true, 'Previous configuration detection must not enable a different dataset');
snapshot('postprocessing', {postprocess:{stage:'mesh',total_episodes:2,completed_episodes:1,
  episode_index:1,elapsed_s:12,streams:{photon:{stage:'mesh',completed_frames:3,total_frames:6}}}});
assert.equal(elements.get('launch').disabled, true);
assert.equal(elements.get('exit').disabled, true);
assert.equal(elements.get('postprocessProgress').classList.contains('hidden'), false);
assert.equal(elements.get('postprocessBar').max, 2);
assert.equal(elements.get('postprocessBar').value, 1);
assert.match(elements.get('postprocessDetail').textContent, /已转换 1 \/ 2 条.*Mesh3DFlow/);
assert.match(elements.get('postprocessStreams').children.at(-1).textContent, /photon.*3 \/ 6 帧/);
snapshot('finished', {has_unsaved:false,postprocess:{stage:'complete',total_episodes:2,completed_episodes:2}});
assert.equal(elements.get('postprocessBar').value, 2);
assert.equal(elements.get('postprocessProgress').classList.contains('hidden'), false);
snapshot('finished', {has_unsaved:false,postprocess:{stage:'complete',total_episodes:0,completed_episodes:0}});
assert.equal(elements.get('postprocessBar').value, 1);
assert.match(elements.get('postprocessDetail').textContent, /暂无待处理数据.*处理完成/);
console.log('Postprocessing checks passed: saved configuration, dataset isolation, mutual exclusion, stage and frame progress.');

// Rebuild confirmation shows the inspected path and never asks the user to type it.
(async () => {
  assert.doesNotMatch(html, /id="confirmRoot"|输入完整数据集路径/);
  assert.doesNotMatch(html, /web-dataset-trash|旧数据移入回收目录/);
  vm.runInContext(`
    selected={path:'tasks/a.yaml',revision:'rev-a',text:'config-a'};loadedText=selected.text;
    $('yamlEditor').value=loadedText;$('configPath').value=selected.path;
  `, context);
  let dataset = {root:'/datasets/任务 a',exists:true,resumable:true,episodes:2};
  const starts = [], confirmations = [];
  let accepted = false;
  context.confirm = message => { confirmations.push(message); return accepted; };
  context.fetch = async (url, options) => {
    if (url === '/api/preflight') return {ok:true,json:async()=>({ticket:'launch-ticket',dataset})};
    assert.equal(url, '/api/start');
    starts.push(JSON.parse(options.body));
    return {ok:true,json:async()=>({state:{phase:'initializing'},controller:true,client_id:'owner',effective:null})};
  };
  const run = code => vm.runInContext(code, context);
  function choose(mode) {
    for (const label of elements.get('datasetChoices').children) {
      const radio = label.children[0];
      radio.checked = radio.value === mode;
      if (radio.checked) radio.onchange();
    }
  }
  await run('launch()');
  choose('rebuild');
  await run('confirmLaunch()');
  assert.equal(starts.length, 0, 'Cancelling the second confirmation must not start or rebuild');
  assert.equal(elements.get('launchDialog').open, true, 'Cancellation must keep launch choices available');
  assert.equal(elements.get('confirmLaunch').disabled, false);
  assert.ok(confirmations[0].includes(dataset.root), 'Confirmation must show the full inspected path');
  assert.match(confirmations[0], /旧数据将被删除/, 'Confirmation must describe overwriting the old dataset');
  accepted = true;
  await run('confirmLaunch()');
  assert.equal(starts.length, 1);
  assert.equal(starts[0].dataset_mode, 'rebuild');
  assert.equal(starts[0].confirm_root, dataset.root, 'Send the inspected path automatically after approval');
  assert.equal(starts[0].ticket, 'launch-ticket');
  assert.equal(elements.get('launchDialog').open, false);
  assert.equal(elements.get('confirmLaunch').disabled, false);

  dataset = {...dataset, root:'/datasets/other-task'};
  await run('launch()');
  choose('rebuild');
  await run('confirmLaunch()');
  assert.ok(confirmations.at(-1).includes(dataset.root), 'A later launch must confirm its own path');
  assert.equal(starts.at(-1).confirm_root, dataset.root);
  const count = confirmations.length;
  await run('launch()');
  choose('resume');
  await run('confirmLaunch()');
  assert.equal(starts.at(-1).dataset_mode, 'resume');
  dataset = {...dataset, exists:false, resumable:false, episodes:0};
  await run('launch()');
  await run('confirmLaunch()');
  assert.equal(starts.at(-1).dataset_mode, 'new');
  assert.equal(confirmations.length, count, 'Resume and new launches must not show a rebuild confirmation');
  console.log('Launch checks passed: rebuild path display, cancellation, automatic path confirmation, resume and new.');
})().catch(error => { console.error(error); process.exitCode = 1; });
