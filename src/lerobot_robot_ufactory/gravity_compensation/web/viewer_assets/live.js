// Runs inside the shared viewer closure: poses use the same URDF transforms.
document.title='GELLO · 实时标定对照';
document.querySelector('header h1 span').textContent='/ 实时标定对照';
document.querySelector('header small').textContent='移动实物 GELLO，观察模型姿态是否一致';
const chip=document.querySelector('.chip');chip.textContent='等待电机数据';chip.setAttribute('role','status');
const style=document.createElement('style');
style.textContent=`
main{grid-template-columns:390px 1fr}body:not(.offline-view) #demo-section,body:not(.offline-view) #angle-section input,body:not(.offline-view) #angle-section .scale,body:not(.offline-view) #angle-section .row,body:not(.offline-view) #reset{display:none}
body.offline-view .live-panel,body.offline-view .live-angles-section{display:none}
.live-table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}.live-table th,.live-table td{padding:7px 4px;text-align:right;border-bottom:1px solid #30404f}.live-table th:first-child,.live-table td:first-child{text-align:left}.live-table th{color:#a7b9ca;font-weight:500}.live-table tr{cursor:pointer}.live-table tr.selected{background:#334431}.live-table .model-angle{color:#f8b052}.live-detail{font-size:12px;overflow-wrap:anywhere}.live-muted{color:#8ca1b4}.live-metric{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;margin:8px 0}.live-metric strong{color:#b7f0e0}.live-alert{color:#ffba99;font-size:12px;overflow-wrap:anywhere;min-height:20px}.live-panel{margin-bottom:20px;border-bottom:1px solid #30404f;padding-bottom:16px}.live-note{font-size:12px;color:#a7b9ca}.live-table-wrap{overflow:auto}
@media(max-width:760px){main{display:flex}.live-table{font-size:12px}}
`;
document.head.append(style);
const panel=document.createElement('div');panel.className='live-panel';
panel.innerHTML=`<h2>实物 → 标定模型</h2><div class="live-metric"><span>采样 <strong id="read-hz">—</strong></span><span>数据年龄 <strong id="sample-age">—</strong></span></div><div class="live-alert" id="live-error" role="status">正在连接电机…</div><div class="row"><button id="freeze">暂停画面</button><button id="live-capture">记录参考姿态</button><button id="download-pose">保存当前读数</button></div><p class="live-note" id="live-mode">实时跟随 · 只读取电机，拖动视角不影响实物。</p>`;
document.querySelector('aside').prepend(panel);
const angles=document.createElement('div');angles.className='section live-angles-section';
angles.innerHTML=`<h2>七轴角度 · 度</h2><div class="live-table-wrap"><table class="live-table"><thead><tr><th>关节</th><th>电机原始</th><th>模型</th><th>参考变化</th></tr></thead><tbody id="live-angles"></tbody></table></div><p class="live-note">模型 = (电机 − 零位) × 方向。参考变化为记录姿态后的最短旋转差，不代表实物误差。</p><p class="live-detail" id="gripper-readout"></p>`;
$('angle-section').after(angles);
const details=document.createElement('div');details.className='section live-detail';
details.innerHTML=`<h2>当前标定</h2><div id="calibration"></div><p class="live-muted" id="profile-path"></p><p class="live-muted" id="model-source"></p>`;
document.querySelector('aside').append(details);
$('profile-path').textContent=`配置：${MODEL.live.profile_path}`;
$('model-source').textContent=`USB ${MODEL.live.usb_serial} · ${MODEL.live.baudrate} baud · URDF ${MODEL.urdf} · SHA256 ${MODEL.urdf_sha256}`;
$('calibration').textContent=`零位(°)：[${MODEL.live.encoder_zero_rad.map(x=>THREE.MathUtils.radToDeg(x).toFixed(2)).join(', ')}]　方向：[${MODEL.live.model_signs.map(x=>x>0?'+1':'−1').join(', ')}]`;
document.querySelector('.stage-title small').textContent='模型按当前零位和方向跟随电机；基座坐标：X 红 / Y 绿 / Z 蓝';
document.querySelector('.legend').lastElementChild.textContent='灰蓝影子：手动记录的参考姿态';
$('transparent').checked=false;$('ghost').checked=false;appearance();
// Until the first valid reading, show the floor/axes but no invented pose.
robot.root.visible=false;axisGroup.visible=false;$('labels').style.display='none';
$('angle-value').textContent='—';$('rad-value').textContent='等待真实电机读数';$('pose').textContent='尚未收到电机数据';
let frozen=false,lastPacket=null,lastSequence=-1,referenceCaptured=false,displayedSample=null;
let lastResponseAt=0,lastDataAge=Infinity,liveStatus='connecting';
let displayedViewMode=null;
function showViewMode(mode){
 if(displayedViewMode!==mode){
  stop();frozen=false;$('freeze').textContent='暂停画面';lastSequence=-1;
  if(mode==='offline'){q=MODEL.initial_q.slice();remember();sync();fit();displayedSample=null;referenceCaptured=false;$('ghost').checked=true;appearance();}
  else if(mode==='read_only'||mode==='compensation'){displayedSample=null;referenceCaptured=false;$('ghost').checked=false;appearance();}
 }
 MODEL.live.mode=mode;
 displayedViewMode=mode;
 const offline=mode==='offline';document.body.classList.toggle('offline-view',offline);
 if(offline){robot.root.visible=true;axisGroup.visible=true;$('labels').style.display='';sync();}
 else if(!displayedSample){robot.root.visible=false;axisGroup.visible=false;$('labels').style.display='none';$('angle-value').textContent='—';$('rad-value').textContent='等待真实电机读数';$('pose').textContent='尚未收到电机数据';}
 document.querySelector('.stage-title small').textContent=offline?'离线演示角度，未读取电机；基座坐标：X 红 / Y 绿 / Z 蓝':'模型按当前零位和方向跟随电机；基座坐标：X 红 / Y 绿 / Z 蓝';
 updateLiveNote();
}
function updateLiveNote(){
 const active=lastPacket?.tuning?.state==='active';
 $('live-mode').textContent=active?(frozen?'画面已暂停 · 补偿仍在出力；停止请点击上方“立即卸力”。':'实时跟随 · 补偿继续运行；停止请点击上方“立即卸力”。'):frozen?'画面已暂停 · 参考及保存使用画面中的姿态。':MODEL.live.mode==='read_only'?'实时跟随 · 只读取电机，拖动视角不影响实物。':'已停止 · 显示最后姿态。';
}
showViewMode('offline');
const rows=MODEL.joints.map((joint,i)=>{
 const row=document.createElement('tr');row.innerHTML=`<td>J${i+1}</td><td>—</td><td class="model-angle">—</td><td>—</td>`;row.onclick=()=>choose(i);$('live-angles').append(row);return row;
});
function fresh(){return liveStatus==='connected'&&lastDataAge+performance.now()-lastResponseAt<=500;}
function showSample(sample){
 const firstSample=displayedSample===null;
 displayedSample=sample;q=sample.model_q_rad.slice();sync();robot.root.visible=true;axisGroup.visible=true;$('labels').style.display='';
 if(firstSample)fit();
 rows.forEach((row,i)=>{
  row.children[1].textContent=THREE.MathUtils.radToDeg(sample.encoder_rad[i]).toFixed(2)+'°';
  row.children[2].textContent=THREE.MathUtils.radToDeg(q[i]).toFixed(2)+'°';
  const delta=Math.atan2(Math.sin(q[i]-referenceQ[i]),Math.cos(q[i]-referenceQ[i]));
  row.children[3].textContent=referenceCaptured?THREE.MathUtils.radToDeg(delta).toFixed(2)+'°':'—';
 });
 if(sample.encoder_rad.length>7)$('gripper-readout').textContent=`ID8 夹爪原始角度：${THREE.MathUtils.radToDeg(sample.encoder_rad[7]).toFixed(2)}°（七轴模型不包含夹爪运动）`;
}
$('freeze').onclick=()=>{
 frozen=!frozen;$('freeze').textContent=frozen?'恢复实时跟随':'暂停画面';
 updateLiveNote();
 if(!frozen&&fresh()&&lastPacket?.sample)showSample(lastPacket.sample);
};
$('live-capture').onclick=()=>{
 if(!displayedSample)return;
 remember();referenceCaptured=true;$('ghost').checked=true;appearance();showSample(displayedSample);
};
$('download-pose').onclick=()=>{
 if(!displayedSample)return;
 const record={captured_at:new Date().toISOString(),profile:MODEL.live,urdf_sha256:MODEL.urdf_sha256,frozen,connection_status:liveStatus,...displayedSample};
 const url=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)],{type:'application/json'}));
 const link=document.createElement('a');link.href=url;link.download=`gello-pose-${new Date().toISOString().replace(/[:.]/g,'-')}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
async function poll(){
 try{
  const response=await fetch('/api/state',{cache:'no-store',signal:AbortSignal.timeout(2000)});
  if(!response.ok)throw Error(`HTTP ${response.status}`);
  const packet=await response.json();
  lastPacket=packet;liveStatus=packet.status;lastResponseAt=performance.now();lastDataAge=packet.age_ms??Infinity;
  showViewMode(packet.view_mode);
  window.dispatchEvent(new CustomEvent('gello-state',{detail:packet}));
  if(packet.view_mode!=='offline'&&packet.status==='connected'&&packet.sample&&packet.sequence!==lastSequence){
   if(packet.sample.model_q_rad.length!==7||!packet.sample.model_q_rad.every(Number.isFinite))throw Error('模型角度无效');
   lastSequence=packet.sequence;if(!frozen)showSample(packet.sample);
  }
  $('live-error').textContent=packet.error||(packet.status==='connecting'?'正在连接电机…':packet.status==='stale'?'读数超时，模型保留最后姿态。':'');
 }catch(error){liveStatus='error';$('live-error').textContent='连接中断：'+error.message+'；保留最后姿态，正在重试。';}
 setTimeout(poll,33);
}
setInterval(()=>{
 const connected=fresh(),age=lastDataAge+performance.now()-lastResponseAt;
 chip.textContent=liveStatus==='error'?'连接异常 · 检查服务':MODEL.live.mode==='offline'?'离线模型 · 不连接电机':connected?(frozen?'电机在线 · 画面暂停':'电机在线 · 实时跟随'):liveStatus==='stopped'?'已卸力 · 等待启动':liveStatus==='connecting'?'正在连接电机':displayedSample?'数据中断 · 最后姿态':'未连接 · 等待读数';
 chip.style.color=connected?'#8ce4cd':'#ffba99';
 $('sample-age').textContent=Number.isFinite(age)?`${Math.round(age)} ms`:'—';
 $('read-hz').textContent=connected&&lastPacket?.sample?.read_hz?`${lastPacket.sample.read_hz.toFixed(1)} Hz`:'—';
 rows.forEach((row,i)=>row.classList.toggle('selected',i===selected));
 $('live-capture').disabled=!displayedSample;$('download-pose').disabled=!displayedSample;
 if(!connected&&liveStatus==='connected')$('live-error').textContent='读数超时，模型保留最后姿态。';
},100);
poll();
