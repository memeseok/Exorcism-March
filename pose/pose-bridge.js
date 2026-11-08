// Camera capture + pose worker. Unity pulls raw landmark frames through ExoBridge.jslib;
// every gesture decision (punch, guard, X, calibration) is made in C# so the editor,
// the keyboard path and the web build all share one detector.
(function () {
  'use strict';
  var BASE = new URL('pose/', document.baseURI).href;
  var LM = 33;
  var K = 4 + LM * 4 + LM * 3; // [t, aspect, hasPose, costMs, lm(x,y,z,vis)*33, world(x,y,z)*33]
  var T0 = performance.now();
  var REC_SECONDS = 90;

  var S = {
    state: 'idle', message: '', label: '', deviceId: '', camCount: 0, camIndex: 0,
    delegate: '', model: '', width: 0, height: 0, ready: false, busy: false, busySince: 0,
    queue: [], video: null, stream: null, worker: null, costAvg: 0,
    camFrames: 0, results: 0, poses: 0, fpsCam: 0, fpsPose: 0, statT: 0, statCam: 0, statPose: 0,
    thumb: null, thumbCtx: null, thumbAt: 0, rec: [], marks: [], attempt: 0
  };
  var LADDER = [['GPU', 'full'], ['CPU', 'full'], ['CPU', 'lite']];

  function setState(s, msg) { S.state = s; S.message = msg || ''; }

  function openCamera(deviceId) {
    var video = { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60, max: 60 } };
    if (deviceId) video.deviceId = { exact: deviceId }; else video.facingMode = 'user';
    return navigator.mediaDevices.getUserMedia({ video: video, audio: false });
  }

  function fail(e) {
    var n = e && e.name;
    if (n === 'NotAllowedError' || n === 'SecurityError') setState('denied', String(e.message || n));
    else if (n === 'NotFoundError' || n === 'OverconstrainedError') setState('nocamera', String(e.message || n));
    else if (n === 'NotReadableError' || n === 'AbortError') setState('busy', String(e.message || n));
    else setState('error', String((e && e.message) || e));
  }

  async function start(preferredId) {
    if (S.state === 'requesting' || S.state === 'loading' || S.state === 'running') return;
    if (!window.isSecureContext) { setState('insecure', 'camera needs https or localhost'); return; }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { setState('unsupported', 'getUserMedia missing'); return; }
    setState('requesting');
    var stream = null;
    try { stream = await openCamera(preferredId || ''); }
    catch (e) {
      if (preferredId) { try { stream = await openCamera(''); } catch (e2) { fail(e2); return; } }
      else { fail(e); return; }
    }
    attachStream(stream);
    if (!S.worker) startWorker(0);
    else if (S.ready) setState('running');
  }

  function stopStream() {
    if (S.stream) { S.stream.getTracks().forEach(function (t) { t.onended = null; t.stop(); }); }
    S.stream = null;
  }

  function attachStream(stream) {
    stopStream();
    S.stream = stream;
    var track = stream.getVideoTracks()[0];
    var st = track.getSettings ? track.getSettings() : {};
    S.deviceId = st.deviceId || '';
    S.label = track.label || '';
    track.onended = function () { setState('lost', 'camera disconnected'); };
    if (!S.video) {
      S.video = document.createElement('video');
      S.video.muted = true; S.video.playsInline = true; S.video.autoplay = true;
      S.video.setAttribute('playsinline', '');
      // Kept in the DOM (nearly invisible) so frame callbacks keep firing in every browser.
      S.video.style.cssText = 'position:fixed;left:0;top:0;width:2px;height:2px;opacity:0.01;pointer-events:none;z-index:-1';
      document.body.appendChild(S.video);
    }
    S.video.srcObject = stream;
    var p = S.video.play(); if (p && p.catch) p.catch(function () {});
    navigator.mediaDevices.enumerateDevices().then(function (list) {
      var cams = list.filter(function (d) { return d.kind === 'videoinput'; });
      S.camCount = cams.length;
      S.camIndex = Math.max(0, cams.findIndex(function (d) { return d.deviceId === S.deviceId; }));
    }).catch(function () {});
    pump(S.video, stream);
  }

  function pump(v, stream) {
    if (v.requestVideoFrameCallback) {
      var cb = function (now, meta) {
        if (S.stream !== stream) return;
        // captureTime should share performance.now()'s clock; anything else falls back to now.
        var ct = meta && meta.captureTime;
        onVideoFrame(ct && Math.abs(ct - performance.now()) < 5000 ? ct : now);
        v.requestVideoFrameCallback(cb);
      };
      v.requestVideoFrameCallback(cb);
    } else {
      var last = -1;
      var loop = function () {
        if (S.stream !== stream) return;
        if (v.currentTime !== last) { last = v.currentTime; onVideoFrame(performance.now()); }
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    }
  }

  function onVideoFrame(captureTime) {
    S.camFrames++;
    var v = S.video;
    if (!v || v.readyState < 2 || !v.videoWidth) return;
    S.width = v.videoWidth; S.height = v.videoHeight;
    updateThumb();
    if (!S.ready) return;
    if (S.busy) {
      // A wedged worker (GPU context loss etc.) is restarted instead of silently freezing input.
      // The first GPU inference compiles shaders and can take seconds; only later stalls demote.
      var limit = S.results < 3 ? 12000 : 2500;
      if (performance.now() - S.busySince > limit) { S.busy = false; startWorker(Math.min(S.attempt + 1, LADDER.length - 1)); }
      return;
    }
    var t = captureTime - T0;
    S.busy = true; S.busySince = performance.now();
    createImageBitmap(v).then(function (bmp) {
      if (!S.worker) { bmp.close(); S.busy = false; return; }
      S.worker.postMessage({ type: 'frame', bitmap: bmp, t: t }, [bmp]);
    }).catch(function () { S.busy = false; });
  }

  function startWorker(step) {
    if (S.worker) { S.worker.terminate(); S.worker = null; }
    S.ready = false; S.busy = false; S.attempt = step; S.costAvg = 0; S.results = 0;
    var cfg = LADDER[step];
    setState('loading', cfg[0] + '/' + cfg[1]);
    var w = new Worker(BASE + 'pose-worker.js');
    S.worker = w;
    w.onmessage = function (e) { if (S.worker === w) onWorker(e.data); };
    w.onerror = function (e) { if (S.worker === w) onWorker({ type: 'init-error', message: String(e.message || 'worker error') }); };
    w.postMessage({ type: 'init', base: BASE, delegate: cfg[0], model: cfg[1] });
  }

  function onWorker(d) {
    if (d.type === 'ready') {
      S.ready = true; S.delegate = d.delegate; S.model = d.model;
      if (S.stream) setState('running');
    } else if (d.type === 'init-error') {
      console.warn('[pose] init failed', d.delegate, d.model, d.message);
      if (S.attempt + 1 < LADDER.length) startWorker(S.attempt + 1);
      else setState('error', d.message);
    } else if (d.type === 'result') {
      S.busy = false;
      S.results++;
      S.costAvg = S.costAvg ? S.costAvg * 0.9 + d.cost * 0.1 : d.cost;
      var f = new Float32Array(K);
      f[0] = d.t; f[1] = S.width / Math.max(1, S.height); f[2] = d.lm ? 1 : 0; f[3] = d.cost;
      if (d.lm) { S.poses++; f.set(d.lm, 4); if (d.wl) f.set(d.wl, 4 + LM * 4); }
      S.queue.push(f);
      if (S.queue.length > 16) S.queue.shift();
      record(f);
      // CPU inference that cannot keep ~25 fps drops to the lite model rather than lagging every punch.
      if (S.attempt < LADDER.length - 1 && S.results > 90 && S.costAvg > (S.delegate === 'GPU' ? 45 : 38)) {
        startWorker(S.attempt + 1);
      }
    }
    var now = performance.now();
    if (now - S.statT > 1000) {
      var dt = (now - S.statT) / 1000;
      S.fpsCam = (S.camFrames - S.statCam) / dt; S.fpsPose = (S.results - S.statPose) / dt;
      S.statT = now; S.statCam = S.camFrames; S.statPose = S.results;
    }
  }

  function updateThumb() {
    var now = performance.now();
    if (now - S.thumbAt < 66) return;
    S.thumbAt = now;
    if (!S.thumb) {
      S.thumb = document.createElement('canvas'); S.thumb.width = 160; S.thumb.height = 120;
      S.thumbCtx = S.thumb.getContext('2d', { willReadFrequently: false });
    }
    try { S.thumbCtx.drawImage(S.video, 0, 0, 160, 120); } catch (_) {}
  }

  function record(f) {
    var a = new Array(K);
    for (var i = 0; i < K; i++) a[i] = Math.round(f[i] * 10000) / 10000;
    S.rec.push(a);
    var cut = f[0] - REC_SECONDS * 1000;
    while (S.rec.length && S.rec[0][0] < cut) S.rec.shift();
    while (S.marks.length && S.marks[0].t < cut) S.marks.shift();
  }

  function saveRecording() {
    var payload = {
      kind: 'exorcist-pose-log', version: 1, savedAt: new Date().toISOString(),
      camera: S.label, model: S.model, delegate: S.delegate, width: S.width, height: S.height,
      layout: 't,aspect,hasPose,costMs,landmarks[x,y,z,vis]*33,world[x,y,z]*33',
      marks: S.marks, frames: S.rec
    };
    var blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'exorcist-pose-' + Date.now() + '.json';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 5000);
  }

  async function nextCamera() {
    try {
      var list = (await navigator.mediaDevices.enumerateDevices()).filter(function (d) { return d.kind === 'videoinput'; });
      if (list.length < 2) return;
      var i = list.findIndex(function (d) { return d.deviceId === S.deviceId; });
      var next = list[(i + 1) % list.length];
      attachStream(await openCamera(next.deviceId));
      if (S.ready) setState('running');
    } catch (e) { fail(e); }
  }

  window.ExoPose = {
    frameSize: K,
    start: function (preferredId) { start(preferredId); },
    drain: function (maxFrames) { var n = Math.min(maxFrames, S.queue.length); return S.queue.splice(0, n); },
    status: function () {
      return JSON.stringify({
        state: S.state, message: S.message, label: S.label, deviceId: S.deviceId,
        camCount: S.camCount, camIndex: S.camIndex, fpsCam: Math.round(S.fpsCam), fpsPose: Math.round(S.fpsPose),
        inferMs: Math.round(S.costAvg * 10) / 10, delegate: S.delegate, model: S.model, width: S.width, height: S.height
      });
    },
    thumbCanvas: function () { return S.thumb; },
    command: function (cmd) {
      if (cmd === 'next-camera') nextCamera();
      else if (cmd === 'rec-save') saveRecording();
      else if (cmd.indexOf('mark:') === 0) S.marks.push({ t: performance.now() - T0, e: cmd.substring(5) });
      else if (cmd.indexOf('start:') === 0) start(cmd.substring(6));
    }
  };
})();
