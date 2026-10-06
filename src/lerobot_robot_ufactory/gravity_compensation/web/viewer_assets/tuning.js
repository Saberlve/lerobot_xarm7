document.title='GELLO · 重力补偿调参';
document.querySelector('header h1 span').textContent='/ 重力补偿调参';
document.querySelector('header small').textContent='逐轴调整支撑，找到容易推动的手感';
const tuneStyle=document.createElement('style');
tuneStyle.textContent=`
main{grid-template-columns:440px 1fr}.tune-toolbar{position:sticky;top:-22px;z-index:8;background:#15222ef8;margin:-22px -22px 16px;padding:20px 22px 16px;border-bottom:1px solid #395368;backdrop-filter:blur(10px)}.tune-toolbar h2{margin-bottom:8px;font-size:18px}.tune-state{font-size:12px;color:#a7b9ca}.tune-actions{display:flex;gap:8px;margin-top:10px}.tune-actions button{flex:1}button.tune-start{background:#68d9ba;color:#0d2923;border-color:#68d9ba;font-weight:700}button.tune-stop{background:#9c343b;border-color:#e88186;color:white;font-weight:700}button:disabled{opacity:.45;cursor:default}.tune-error{color:#ffb49f;font-size:12px;overflow-wrap:anywhere;margin:8px 0 0}.gain-card{background:#1c2d3b;border:1px solid #354c5f;border-radius:10px;padding:11px 13px;margin:9px 0}.gain-card.selected{border-color:#f8b052}.gain-head{display:flex;justify-content:space-between;align-items:center;gap:8px}.gain-head button{padding:3px 7px;border:0;background:none;color:#c4d9e8;font-weight:700}.gain-head input{width:95px;padding:5px 7px;font-size:18px;font-variant-numeric:tabular-nums}.gain-card input[type=range]{margin:9px 0 3px;accent-color:#68d9ba}.gain-readout{font-size:11px;color:#a8becf;display:flex;justify-content:space-between;gap:6px;font-variant-numeric:tabular-nums}.gain-meter{height:3px;background:#344c60;border-radius:2px;margin-top:8px}.gain-meter div{height:100%;background:#68d9ba;border-radius:2px;transition:width .12s}.gain-card.limited{border-color:#e88186}.gain-card.limited .gain-meter div{background:#e88186}.tune-small{font-size:12px;color:#a7b9ca;margin:10px 0}.tune-saving{font-size:12px;color:#84ddc7;min-height:20px;margin:5px 0}.tune-secondary{display:flex;gap:7px;flex-wrap:wrap}.tune-secondary button{font-size:12px;padding:7px 9px}.tune-meta{font-size:11px;color:#90a5b7;overflow-wrap:anywhere;margin-top:12px}
@media(max-width:760px){main{display:flex}aside{order:0}.stage{order:1}.tune-toolbar{top:0;margin-top:0}.gain-card{padding:10px 12px}}
`;
document.head.append(tuneStyle);
tuneStyle.textContent+=`.slew-head{display:flex;justify-content:space-between;align-items:center;margin-top:12px;gap:8px;font-size:12px;color:#b7cad9}.slew-head input{width:80px;padding:4px 6px}.slew-readout{font-size:11px;color:#a8becf;font-variant-numeric:tabular-nums}.thermal-readout{margin-top:6px;font-size:12px;color:#84ddc7}.thermal-readout.hot{color:#ffb49f}.tune-thermal{font-size:12px;color:#a7b9ca;margin-top:8px}.tune-thermal.hot{color:#ffb49f}.slew-presets{margin:10px 0}`;
const tuneToolbar=document.createElement('div');tuneToolbar.className='tune-toolbar';
tuneToolbar.innerHTML=`<h2>模型查看与补偿调参</h2><div class="tune-state" id="tune-state" role="status">离线查看 · 不连接电机</div><div class="tune-secondary"><button id="view-offline">离线模型演示</button><button id="view-read">只读电机角度</button></div><div class="tune-actions"><button class="tune-start" id="tune-start">开始持续补偿</button><button class="tune-stop" id="tune-stop" disabled>立即卸力</button></div><div class="tune-error" id="tune-error" role="alert"></div>`;
const tuneBoard=document.createElement('details');tuneBoard.className='tune-board';
tuneBoard.innerHTML=`<summary>补偿参数 · 可展开预设增益</summary><div class="tune-small">每轴增益独立生效。默认使用已确认的增益，调整时小步修改；增益为 0 时该轴不补重力，仍保留阻尼。</div><div class="tune-secondary"><button id="tune-j2-off">J2 重力补偿归零</button><button id="tune-lower">全部降低 20%</button><button id="tune-defaults">恢复默认增益</button></div><div class="tune-saving" id="tune-saving">等待状态同步</div><div id="gain-cards"></div><div class="row"><button id="tune-export">保存增益配置</button></div><div class="tune-small">持续运行至卸力、故障或页面失联。页面失联超过 3 秒自动卸力；关闭页面或终端退出也会卸力。启动时托住并保持不动，等待 2 秒缓升后再轻推关节。</div><div class="tune-meta" id="tune-meta"></div>`;
tuneStyle.textContent+=`.tune-board>summary{cursor:pointer;color:#b9d4e7;padding:6px 0 12px}`;
let tuneBoardMode='offline';
const aside=document.querySelector('aside');aside.prepend(tuneBoard);aside.prepend(tuneToolbar);
const slewPresets=document.createElement('div');slewPresets.className='tune-small slew-presets';
slewPresets.innerHTML=`<strong>电流跟随速度</strong><p>每轴 50–200 mA/s，数值越大跟随越快。默认使用已确认的逐轴变化率，启动前 2 秒仍用原限速；调整后观察温升和冲击。</p><div class="tune-secondary"><button id="slew-defaults">恢复默认变化率</button><button id="slew-slow">全部降至 50 mA/s</button></div><div id="slew-saving" class="tune-saving">等待变化率同步</div>`;
tuneBoard.insertBefore(slewPresets,$('gain-cards'));
const thermalSummary=document.createElement('div');thermalSummary.id='tune-thermal';thermalSummary.className='tune-thermal';
thermalSummary.textContent='温度 — · 40°C 起提示，45°C 自动卸力';tuneToolbar.append(thermalSummary);
document.querySelector('.live-panel h2').textContent='实时姿态与电机读数';
updateLiveNote();
const tuneConfig=MODEL.tuning;
let draftGains=tuneConfig.initial_gains.slice(),pendingGains=null,sendingGains=false,dirtyGains=false;
let draftSlew=tuneConfig.initial_slew_a_s.slice(),pendingSlew=null,sendingSlew=false,dirtySlew=false;
let tunePhase='idle',tuneBusy=false,lastTunePacket=null;
const gainCards=MODEL.joints.map((joint,i)=>{
 const card=document.createElement('div');card.className='gain-card';
 card.innerHTML=`<div class="gain-head"><button aria-label="观察 J${i+1}">J${i+1}${i===1?' · 当前重点':''}</button><input type="number" min="0" max="1" step="0.005" aria-label="J${i+1} 增益"></div><input type="range" min="0" max="1" step="0.005" aria-label="J${i+1} 增益滑块"><div class="gain-readout"><span class="applied">生效 —</span><span class="command">命令 —</span><span class="measured">实测 —</span></div><div class="gain-meter"><div style="width:0%"></div></div>`;
 $('gain-cards').append(card);
 const slewControls=document.createElement('div');
 slewControls.innerHTML=`<div class="slew-head"><label for="slew-number-${i}">电流变化率 · mA/s</label><input id="slew-number-${i}" class="slew-number" type="number" min="50" max="200" step="10" aria-label="J${i+1} 电流变化率"></div><input class="slew-slider" type="range" min="50" max="200" step="10" aria-label="J${i+1} 电流变化率滑块"><div class="slew-readout">实际限速 — · 电流跟随差 —</div><div class="thermal-readout">温度 —</div>`;
 card.append(slewControls);
 card.querySelector('button').onclick=()=>choose(i);
 const number=card.querySelector('input[type=number]'),slider=card.querySelector('input[type=range]');
 number.classList.add('gain-number');slider.classList.add('gain-slider');
 slider.oninput=()=>{draftGains[i]=Number(slider.value);number.value=draftGains[i].toFixed(3);dirtyGains=true;scheduleGains();};
 slider.onchange=()=>queueGains(draftGains);
 number.onchange=()=>{
  const value=Number(number.value);
  if(!number.value||!Number.isFinite(value)||value<0||value>1){$('tune-error').textContent='增益必须在 0–1 之间';number.value=draftGains[i].toFixed(3);return;}
  draftGains[i]=value;slider.value=value;queueGains(draftGains);
 };
 const slewNumber=card.querySelector('.slew-number'),slewSlider=card.querySelector('.slew-slider');
 slewSlider.oninput=()=>{draftSlew[i]=Number(slewSlider.value)/1000;slewNumber.value=slewSlider.value;dirtySlew=true;scheduleSlew();};
 slewSlider.onchange=()=>queueSlew(draftSlew);
 slewNumber.onchange=()=>{
  const value=Number(slewNumber.value)/1000;
  if(!slewNumber.value||!Number.isFinite(value)||value<tuneConfig.min_slew_a_s||value>tuneConfig.max_slew_a_s){$('tune-error').textContent='电流变化率必须在 50–200 mA/s 之间';slewNumber.value=Math.round(draftSlew[i]*1000);return;}
  draftSlew[i]=value;slewSlider.value=Math.round(value*1000);queueSlew(draftSlew);
 };
 return card;
});
function showDraft(){gainCards.forEach((card,i)=>{
 const number=card.querySelector('input[type=number]');
 if(document.activeElement!==number)number.value=draftGains[i].toFixed(3);
 card.querySelector('input[type=range]').value=draftGains[i];
});}
function showSlew(){gainCards.forEach((card,i)=>{
 const number=card.querySelector('.slew-number');
 if(document.activeElement!==number)number.value=Math.round(draftSlew[i]*1000);
 card.querySelector('.slew-slider').value=Math.round(draftSlew[i]*1000);
});}
showDraft();showSlew();choose(1);
async function controlPost(path,body={}){
 const response=await fetch('/api/'+path,{method:'POST',headers:{'Content-Type':'application/json','X-Gello-Token':tuneConfig.token},body:JSON.stringify(body),signal:AbortSignal.timeout(12000)});
 const result=await response.json();
 if(!response.ok)throw Error(result.error||`HTTP ${response.status}`);
 return result;
}
let gainTimer=null;
function scheduleGains(){clearTimeout(gainTimer);$('tune-saving').textContent='准备应用…';gainTimer=setTimeout(()=>queueGains(draftGains),150);}
async function queueGains(values){
 clearTimeout(gainTimer);dirtyGains=true;pendingGains=values.slice();
 if(sendingGains)return;
 sendingGains=true;
 try{
  while(pendingGains){const next=pendingGains;pendingGains=null;$('tune-saving').textContent='正在应用…';await controlPost('gains',{gains:next});}
  dirtyGains=false;$('tune-saving').textContent=tunePhase==='active'?'增益已应用，电流按限速平滑变化':'增益已设置，下次启动时生效';$('tune-error').textContent='';
 }catch(error){pendingGains=null;dirtyGains=false;$('tune-error').textContent=error.message;$('tune-saving').textContent='增益未能应用，请查看状态';}
 finally{sendingGains=false;}
}
$('tune-j2-off').onclick=()=>{draftGains[1]=0;showDraft();queueGains(draftGains);};
$('tune-lower').onclick=()=>{draftGains=draftGains.map(x=>Math.round(x*.8*1000)/1000);showDraft();queueGains(draftGains);};
$('tune-defaults').onclick=()=>{draftGains=tuneConfig.initial_gains.slice();showDraft();queueGains(draftGains);};
let slewTimer=null;
function scheduleSlew(){clearTimeout(slewTimer);$('slew-saving').textContent='准备应用…';slewTimer=setTimeout(()=>queueSlew(draftSlew),150);}
async function queueSlew(values){
 clearTimeout(slewTimer);dirtySlew=true;pendingSlew=values.slice();
 if(sendingSlew)return;
 sendingSlew=true;
 try{
  while(pendingSlew){const next=pendingSlew;pendingSlew=null;$('slew-saving').textContent='正在应用…';await controlPost('slew',{current_slew_a_s:next});}
  dirtySlew=false;$('slew-saving').textContent=tunePhase==='active'?'运行变化率已设置，启动缓升期间保持原限速':'变化率已设置，下次启动时生效';$('tune-error').textContent='';
 }catch(error){pendingSlew=null;dirtySlew=false;$('tune-error').textContent=error.message;$('slew-saving').textContent='变化率未能应用，请查看状态';}
 finally{sendingSlew=false;}
}
$('slew-slow').onclick=()=>{draftSlew=Array(7).fill(.05);showSlew();queueSlew(draftSlew);};
$('slew-defaults').onclick=()=>{draftSlew=tuneConfig.initial_slew_a_s.slice();showSlew();queueSlew(draftSlew);};
function updateControls(){
 const changing=tuneBusy||tunePhase==='starting'||tunePhase==='stopping';
 $('tune-start').disabled=changing||tunePhase==='active'||sendingGains||sendingSlew||dirtyGains||dirtySlew;
 $('tune-start').textContent=tunePhase==='starting'?'正在启动…':tunePhase==='fault'?'重新启动补偿':'开始持续补偿';
 $('tune-stop').disabled=!['active','starting','stopping'].includes(tunePhase)&&!tuneBusy;
 $('tune-stop').disabled=$('tune-stop').disabled&&tunePhase!=='reading';
 $('tune-stop').textContent=tunePhase==='reading'?'停止读取':'立即卸力';
 for(const id of ['view-offline','view-read'])$(id).disabled=changing||tunePhase==='active';
 $('view-offline').classList.toggle('active',MODEL.live.mode==='offline');
 $('view-read').classList.toggle('active',MODEL.live.mode==='read_only');
 gainCards.forEach((card,i)=>{card.classList.toggle('selected',i===selected);card.querySelectorAll('input').forEach(input=>input.disabled=changing);});
 for(const id of ['tune-j2-off','tune-lower','tune-defaults','slew-slow','slew-defaults'])$(id).disabled=changing;
}
for(const [id,mode] of [['view-offline','offline'],['view-read','read_only']])$(id).onclick=async()=>{
 tuneBusy=true;$('tune-error').textContent='';updateControls();
 try{await controlPost('view',{mode});}
 catch(error){$('tune-error').textContent=error.message;}
 finally{tuneBusy=false;}
};
$('tune-start').onclick=async()=>{
 if(sendingGains||sendingSlew||dirtyGains||dirtySlew)return;
 tuneBusy=true;tunePhase='starting';$('tune-error').textContent='';updateControls();
 try{await controlPost('start');}
 catch(error){$('tune-error').textContent=error.message;}
 finally{tuneBusy=false;}
};
$('tune-stop').onclick=async()=>{
 tunePhase='stopping';$('tune-state').textContent='正在关闭扭矩…';updateControls();
 try{await controlPost('stop');}
 catch(error){$('tune-error').textContent=error.message;}
};
window.addEventListener('gello-state',event=>{
 const packet=event.detail;if(!packet.tuning)return;
 if(tuneBoardMode!==packet.view_mode){tuneBoardMode=packet.view_mode;if(packet.view_mode==='compensation')tuneBoard.open=true;else if(['offline','read_only'].includes(packet.view_mode))tuneBoard.open=false;}
 lastTunePacket=packet;tunePhase=packet.tuning.state;
 const labels={idle:'离线查看 · 不连接电机',reading:'只读角度 · 不启用扭矩',starting:'正在切换 · 启动补偿时请托住并保持不动',active:'补偿运行中 · 增益调节实时生效',stopping:'正在停止并释放串口…',stopped:'已停止 · 可以只读查看或启动补偿',fault:'保护停机 · 请查看故障原因'};
 $('tune-state').textContent=labels[tunePhase]||tunePhase;
 $('tune-state').style.color=tunePhase==='active'?'#84ddc7':tunePhase==='fault'?'#ffb49f':'#a7b9ca';
 if(packet.error)$('tune-error').textContent=packet.error;
 if(!dirtyGains&&!sendingGains){draftGains=packet.tuning.gains.slice();showDraft();}
 if(!dirtySlew&&!sendingSlew){draftSlew=packet.tuning.current_slew_a_s.slice();showSlew();}
 if($('slew-saving').textContent==='等待变化率同步')$('slew-saving').textContent='已同步逐轴变化率';
 if($('tune-saving').textContent==='等待状态同步')$('tune-saving').textContent='逐轴增益已同步';
 const record=packet.tuning.record,active=tunePhase==='active';
 const temperatures=packet.view_mode==='read_only'?packet.sample?.temperature_c:record?.temperature_c;
 const temperatureNote=active?'':packet.view_mode==='read_only'?(packet.status==='connected'?'':' · 最后读数'):' · 停机时读数';
 gainCards.forEach((card,i)=>{
  card.querySelector('.applied').textContent=active&&packet.tuning.applied_gains?`生效 ${packet.tuning.applied_gains[i].toFixed(3)}`:'待启动';
  card.querySelector('.command').textContent=active&&record?`命令 ${(record.target_a[i]*1000).toFixed(0)} mA`:'命令 —';
  card.querySelector('.measured').textContent=active&&record?`实测 ${(record.measured_current_a[i]*1000).toFixed(0)} mA`:'实测 —';
  card.querySelector('.gain-meter div').style.width=active&&record?`${Math.min(100,Math.abs(record.target_a[i])/packet.tuning.current_limit_a[i]*100)}%`:'0%';
  card.classList.toggle('limited',active&&record&&Math.abs(record.requested_a[i])>packet.tuning.current_limit_a[i]);
  card.querySelector('.slew-readout').textContent=active&&record?`实际限速 ${(record.current_slew_a_s[i]*1000).toFixed(0)} mA/s · 跟随差 ${(Math.abs(record.slew_error_a[i])*1000).toFixed(0)} mA`:'实际限速 — · 电流跟随差 —';
  const temp=card.querySelector('.thermal-readout');
  temp.textContent=temperatures?`温度 ${temperatures[i]}°C${temperatureNote}`:'温度 —';
  temp.classList.toggle('hot',!!temperatures&&temperatures[i]>=40);
 });
 const hottest=temperatures?Math.max(...temperatures):null;
 thermalSummary.textContent=`${hottest===null?'温度 —':`最高 ${hottest}°C${temperatureNote}`} · 补偿时 ${packet.tuning.temperature_limit_c}°C 自动卸力`;
 thermalSummary.classList.toggle('hot',hottest!==null&&hottest>=40);
 $('tune-meta').textContent=`每轴限流 ${packet.tuning.current_limit_a.map(x=>(x*1000).toFixed(0)).join('/')} mA · 持续模式不限制相对启动姿态的位移${packet.tuning.log?' · 日志 '+packet.tuning.log:''}`;
 updateControls();
});
async function tuneHeartbeat(){
 try{await controlPost('heartbeat');}
 catch(error){$('tune-error').textContent='页面与服务连接失败：'+error.message;}
 setTimeout(tuneHeartbeat,700);
}
tuneHeartbeat();
window.addEventListener('pagehide',()=>{
 fetch('/api/stop',{method:'POST',headers:{'Content-Type':'application/json','X-Gello-Token':tuneConfig.token},body:'{}',keepalive:true}).catch(()=>{});
});
$('tune-export').onclick=()=>{
 const gains=lastTunePacket?.tuning?.gains||draftGains;
 const slew=lastTunePacket?.tuning?.current_slew_a_s||draftSlew;
 const yaml=`gravity_compensation:\n  enabled: true\n  experimental: true\n  profile_path: ${JSON.stringify(MODEL.live.profile_path)}\n  gain: 0.3\n  joint_gains: [${gains.map(x=>Number(x.toFixed(4))).join(', ')}]\n  running_current_slew_a_s: [${slew.join(', ')}]\n  j5_gain: null\n  j6_gain: null\n`;
 const url=URL.createObjectURL(new Blob([yaml],{type:'text/yaml'}));
 const link=document.createElement('a');link.href=url;link.download='gello-gravity-gains.yaml';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
setInterval(updateControls,100);
