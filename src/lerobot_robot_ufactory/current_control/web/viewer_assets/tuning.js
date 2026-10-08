;(() => {
document.title='GELLO · 恒流与阻尼';
document.querySelector('header h1 span').textContent='/ 恒流与阻尼';
document.querySelector('header small').textContent='模型查看与独立电流控制';
const tuneConfig=MODEL.tuning;
const board=document.createElement('section');
board.innerHTML=`<h2>恒流与阻尼</h2><p id="current-state">离线 · 未启用电机</p><div class="row"><button id="current-offline">离线模型</button><button id="current-read">只读角度</button></div><div class="row"><button id="current-start">启动恒流与阻尼</button><button id="current-stop">立即卸力</button></div><p id="current-error" role="alert"></p><p>逐轴修改恒流与阻尼后点击“应用并同步配置”。运行时按电流变化率过渡；未启动时仅设置下次启动值。电流变化率修改后自动同步。恒流正负表示电机出力方向；阻尼始终反向于运动，速度低于 ${MODEL.tuning.damping_deadband_rad_s} rad/s 时为零。启动时托住手臂，等待 2 秒缓升。</p><div id="current-cards"></div><div class="row"><button id="current-save">重新同步参数到配置</button></div><p id="current-saved" role="status"></p><p id="current-temperature"></p><p>调参完成后立即卸力，再启动遥操作。遥操作在下次启动时读取同步后的恒流、阻尼与电流变化率。页面失联超过 3 秒自动卸力。</p>`;
document.querySelector('aside').prepend(board);
let phase='idle',busy=false,lastPacket=null;
const cards=MODEL.joints.map((joint,i)=>{
 const card=document.createElement('div');card.style.cssText='padding:12px;border-bottom:1px solid #395368';
 const fixed=tuneConfig.constant_current_a[i]*1000,damping=tuneConfig.constant_damping_a[i]*1000,limit=tuneConfig.current_limit_a[i]*1000;
 card.innerHTML=`<strong>J${i+1}</strong><p class="reading">命令 — · 实测 —</p><div><label>恒流 mA <input class="constant" aria-label="J${i+1} 恒流 mA" type="number" min="${-limit}" max="${limit}" step="1" required value="${fixed}" style="width:80px"></label></div><div><label>阻尼 mA <input class="damping" aria-label="J${i+1} 阻尼 mA" type="number" min="0" max="${limit}" step="1" required value="${damping}" style="width:80px"></label></div><button class="apply">应用并同步配置</button><div><label>运行电流变化率 mA/s <input class="slew" aria-label="J${i+1} 电流变化率 mA/s" type="number" min="50" max="200" step="1" required value="${tuneConfig.initial_slew_a_s[i]*1000}" style="width:80px"></label></div><p class="thermal">温度 —</p>`;
 card.querySelectorAll('.constant,.damping').forEach(input=>input.oninput=()=>{card.dataset.draft='true';});
 card.querySelector('.apply').onclick=async()=>{
  const fixedInput=card.querySelector('.constant'),dampingInput=card.querySelector('.damping');
  if(!fixedInput.reportValidity()||!dampingInput.reportValidity())return;
  const constant=Number(fixedInput.value)/1000,damping=Number(dampingInput.value)/1000;
  if(Math.abs(constant)+damping>tuneConfig.current_limit_a[i]){$('current-error').textContent=`J${i+1} 恒流绝对值与阻尼之和不能超过 ${limit} mA`;return;}
  const currents=(lastPacket?.tuning.constant_current_a||tuneConfig.constant_current_a).slice();
  const dampings=(lastPacket?.tuning.constant_damping_a||tuneConfig.constant_damping_a).slice();
  currents[i]=constant;dampings[i]=damping;
  const packet=await action('currents',{constant_current_a:currents,constant_damping_a:dampings});
  const applied=packet||lastPacket;
  if(applied?.tuning.constant_current_a[i]===constant&&applied?.tuning.constant_damping_a[i]===damping)delete card.dataset.draft;
 };
 card.querySelector('.slew').onchange=async event=>{
  if(!event.target.reportValidity())return;
  const value=Number(event.target.value)/1000;
  if(!Number.isFinite(value)||value<.05||value>.2){$('current-error').textContent='变化率范围为 50–200 mA/s';return;}
  const values=(lastPacket?.tuning.current_slew_a_s||tuneConfig.initial_slew_a_s).slice();values[i]=value;
  await action('slew',{current_slew_a_s:values});
 };
 $('current-cards').append(card);return card;
});
async function post(path,body={}){
 const response=await fetch('/api/'+path,{method:'POST',headers:{'Content-Type':'application/json','X-Gello-Token':tuneConfig.token},body:JSON.stringify(body),signal:AbortSignal.timeout(12000)});
 const result=await response.json();if(!response.ok)throw Error(result.error||`HTTP ${response.status}`);return result;
}
function controls(){
 const changing=busy||['starting','stopping'].includes(phase);
 $('current-start').disabled=changing||phase==='active';
 $('current-stop').disabled=!busy&&!['active','reading','starting','stopping'].includes(phase);
 $('current-offline').disabled=$('current-read').disabled=changing||phase==='active';
 $('current-save').disabled=changing||!lastPacket?.tuning.unsaved_settings||cards.some(c=>c.dataset.draft);
 cards.forEach(c=>{c.querySelectorAll('input,button').forEach(input=>input.disabled=changing);});
}
async function action(path,body={}){
 busy=true;controls();$('current-error').textContent='';
 try{const packet=await post(path,body);window.dispatchEvent(new CustomEvent('gello-state',{detail:packet}));return packet;}
 catch(e){
  $('current-error').textContent=e.message;
  // An apply can succeed while saving fails. Refresh its status so retry stays available.
  try{const response=await fetch('/api/state',{cache:'no-store',signal:AbortSignal.timeout(2000)});if(response.ok)window.dispatchEvent(new CustomEvent('gello-state',{detail:await response.json()}));}catch{}
 }
 finally{busy=false;controls();}
}
$('current-start').onclick=()=>action('start');
$('current-stop').onclick=()=>action('stop');
$('current-offline').onclick=()=>action('view',{mode:'offline'});
$('current-read').onclick=()=>action('view',{mode:'read_only'});
$('current-save').onclick=()=>action('save-settings');
window.addEventListener('gello-state',event=>{
 const packet=event.detail;if(!packet.tuning)return;lastPacket=packet;phase=packet.tuning.state;
 const labels={idle:'离线 · 未启用电机',reading:'只读角度 · 未启用扭矩',starting:'正在启动，请托住手臂',active:'恒流与阻尼运行中',stopping:'正在卸力',stopped:'已停止',fault:'保护停机'};
 $('current-state').textContent=labels[phase]||phase;
 if(packet.error)$('current-error').textContent=packet.error;
 const record=packet.tuning.record,active=phase==='active';
 const temps=packet.view_mode==='read_only'?packet.sample?.temperature_c:record?.temperature_c;
 cards.forEach((card,i)=>{
  card.querySelector('.reading').textContent=active&&record?`命令 ${(record.target_a[i]*1000).toFixed(0)} mA · 实测 ${(record.measured_current_a[i]*1000).toFixed(0)} mA`:'命令 — · 实测 —';
  card.querySelector('.thermal').textContent=temps?`温度 ${temps[i]}°C${active||packet.view_mode==='read_only'?'':'（最后读数）'}`:'温度 —';
  if(!busy){
   for(const [selector,values] of [['.constant',packet.tuning.constant_current_a],['.damping',packet.tuning.constant_damping_a],['.slew',packet.tuning.current_slew_a_s]]){
    const input=card.querySelector(selector);
    if(document.activeElement!==input&&(selector==='.slew'||!card.dataset.draft))input.value=Math.round(values[i]*1000);
   }
  }
 });
 $('current-saved').textContent=cards.some(c=>c.dataset.draft)?'存在尚未应用的输入，请先逐轴应用并同步。':packet.tuning.unsaved_settings?'参数已应用，但尚未同步到配置，请点击重新同步。':'恒流、阻尼和电流变化率已同步到配置；重启遥操作后生效。';
 $('current-temperature').textContent=`达到 ${packet.tuning.temperature_limit_c}°C 自动卸力；电流、通信保护保持启用。`;
 controls();
});
async function heartbeat(){try{await post('heartbeat');}catch(e){$('current-error').textContent='连接失败：'+e.message;}setTimeout(heartbeat,700);}
heartbeat();controls();
window.addEventListener('pagehide',()=>fetch('/api/stop',{method:'POST',headers:{'Content-Type':'application/json','X-Gello-Token':tuneConfig.token},body:'{}',keepalive:true}).catch(()=>{}));

})();
