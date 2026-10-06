Unified GELLO tuning page assets

three.min.js: Three.js 0.160.0, MIT licensed.
Source: https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js
License: THREE_LICENSE.txt

viewer.html contains the shared URDF renderer and offline direction demo.
live.js adds encoder streaming, reference poses and angle readouts.
tuning.js adds read-only connection, compensation lifecycle and tuning controls.
model_web.py embeds all assets, the license and URDF geometry into the single
page served by tuning_web.py. No external asset requests are needed.
