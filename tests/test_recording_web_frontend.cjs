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
    this.classList = { add() {}, remove() {}, toggle() {} };
  }
  append(...items) { this.children.push(...items); }
  replaceChildren() { this.children = []; }
  querySelectorAll() { return []; }
  closest() { return null; }
  focus() {}
  showModal() { this.open = true; }
  close() { this.open = false; }
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
