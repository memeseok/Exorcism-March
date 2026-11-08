// MediaPipe Pose runs here so inference never blocks the Unity main loop.
// Messages in:  {type:'init', base, model:'full'|'lite', delegate:'GPU'|'CPU'}
//               {type:'frame', bitmap:ImageBitmap, t:ms}
// Messages out: {type:'ready'|'init-error'|'result'}
var pose = null;
var lastTs = -1;
var LM = 33;

self.onmessage = function (e) {
  var d = e.data;
  if (d.type === 'init') init(d);
  else if (d.type === 'frame') frame(d);
};

async function init(d) {
  try {
    if (typeof Vision === 'undefined') importScripts(d.base + 'vision_bundle.js');
    var fileset = await Vision.FilesetResolver.forVisionTasks(d.base + 'wasm');
    if (pose) { try { pose.close(); } catch (_) {} pose = null; }
    pose = await Vision.PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: d.base + 'models/pose_landmarker_' + d.model + '.task', delegate: d.delegate },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      // Slightly lower presence/tracking thresholds keep the skeleton through fast, blurry punches.
      minPosePresenceConfidence: 0.4,
      minTrackingConfidence: 0.4
    });
    lastTs = -1;
    self.postMessage({ type: 'ready', delegate: d.delegate, model: d.model });
  } catch (err) {
    self.postMessage({ type: 'init-error', delegate: d.delegate, model: d.model, message: String((err && err.message) || err) });
  }
}

function frame(d) {
  var start = performance.now();
  var lm = null, wl = null, error = '';
  try {
    if (!pose) throw new Error('pose not ready');
    // VIDEO mode requires strictly increasing timestamps, including across camera switches.
    var ts = Math.max(Math.round(d.t), lastTs + 1);
    lastTs = ts;
    var r = pose.detectForVideo(d.bitmap, ts);
    if (r && r.landmarks && r.landmarks.length) {
      var p = r.landmarks[0];
      lm = new Float32Array(LM * 4);
      for (var i = 0; i < LM && i < p.length; i++) {
        lm[i * 4] = p[i].x; lm[i * 4 + 1] = p[i].y; lm[i * 4 + 2] = p[i].z;
        lm[i * 4 + 3] = p[i].visibility === undefined ? 1 : p[i].visibility;
      }
      if (r.worldLandmarks && r.worldLandmarks.length) {
        var w = r.worldLandmarks[0];
        wl = new Float32Array(LM * 3);
        for (var j = 0; j < LM && j < w.length; j++) { wl[j * 3] = w[j].x; wl[j * 3 + 1] = w[j].y; wl[j * 3 + 2] = w[j].z; }
      }
    }
  } catch (err) {
    error = String((err && err.message) || err);
  } finally {
    if (d.bitmap && d.bitmap.close) d.bitmap.close();
  }
  var out = { type: 'result', t: d.t, cost: performance.now() - start, lm: lm, wl: wl, error: error };
  var transfer = [];
  if (lm) transfer.push(lm.buffer);
  if (wl) transfer.push(wl.buffer);
  self.postMessage(out, transfer);
}
