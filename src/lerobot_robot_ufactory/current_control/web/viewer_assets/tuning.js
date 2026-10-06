;(() => {
document.title='GELLO · 恒流与阻尼';
document.querySelector('header h1 span').textContent='/ 恒流与阻尼';
document.querySelector('header small').textContent='模型查看与独立电流控制';
const tuneConfig=MODEL.tuning;
const board=document.createElement('section');
board.innerHTML=`<h2>恒流与阻尼</h2><p id="current-state">离线 · 未启用电机</p><div class="row"><button id="current-offline">离线模型</button><button id="current-read">只读角度</button></div><div class="row"><button id="current-start">启动恒流与阻尼</button><button id="current-stop">立即卸力</button></div><p id="current-error" role="alert"></p><p>当前仅输出下列恒流与阻尼。J3/J7 阻尼始终反向于运动，速度低于 0.05 rad/s 时为零。启动时托住手臂，等待 2 秒缓升。</p><div id="current-cards"></div><p id="current-temperature"></p><p>页面失联超过 3 秒自动卸力。固定电流与阻尼数值保存在服务器配置中。</p>`;
document.querySelector('aside').prepend(board);
let phase='idle',busy=false,lastPacket=null;
const cards=MODEL.joints.map((joint,i)=>{
 const card=document.createElement('div');card.style.cssText='padding:12px;border-bottom:1px solid #395368';
 const fixed=tuneConfig.constant_current_a[i]*1000,damping=tuneConfig.constant_damping_a[i]*1000;
 card.innerHTML=`<strong>J${i+1} · ${damping?`阻尼 ${damping} mA`:`恒流 ${fixed} mA`}</strong><p class="reading">命令 — · 实测 —</p><label>运行电流变化率 mA/s <input type="number" min="50" max="200" step="10" value="${tuneConfig.initial_slew_a_s[i]*1000}" style="width:80px"></label><p class="thermal">温度 —</p>`;
 card.querySelector('input').onchange=async event=>{
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
 cards.forEach(c=>c.querySelector('input').disabled=changing);
}
async function action(path,body={}){
 busy=true;controls();$('current-error').textContent='';
 try{const packet=await post(path,body);window.dispatchEvent(new CustomEvent('gello-state',{detail:packet}));}
 catch(e){$('current-error').textContent=e.message;}
 finally{busy=false;controls();}
}
$('current-start').onclick=()=>action('start');
$('current-stop').onclick=()=>action('stop');
$('current-offline').onclick=()=>action('view',{mode:'offline'});
$('current-read').onclick=()=>action('view',{mode:'read_only'});
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
  const input=card.querySelector('input');if(document.activeElement!==input&&!busy)input.value=Math.round(packet.tuning.current_slew_a_s[i]*1000);
 });
 $('current-temperature').textContent=`达到 ${packet.tuning.temperature_limit_c}°C 自动卸力；电流、通信保护保持启用。`;
 controls();
});
async function heartbeat(){try{await post('heartbeat');}catch(e){$('current-error').textContent='连接失败：'+e.message;}setTimeout(heartbeat,700);}
heartbeat();controls();
window.addEventListener('pagehide',()=>fetch('/api/stop',{method:'POST',headers:{'Content-Type':'application/json','X-Gello-Token':tuneConfig.token},body:'{}',keepalive:true}).catch(()=>{}));

})();
