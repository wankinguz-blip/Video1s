// Video renderer: parallel workers → H.264 segments → concat (+ soundtrack) per screen.
//
//   node scripts/render.mjs --screens left,center,right [--mode standalone|spanned]
//        [--scale 1] [--fps 30] [--workers 3] [--workers-center 2] [--crf 16] [--preset slow]
//        [--out out/video] [--tag NAME]
//
// Frames are sampled at t = i / fps for i = 0 … 15·fps − 1, so playback of the file on repeat is
// exactly periodic (frame 15·fps would be identical to frame 0).
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdirSync, writeFileSync, rmSync, existsSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createCanvas, Path2D } from '@napi-rs/canvas';
import { layoutFor } from '../src/layout.js';
import { SCREENS, LOOP } from '../src/config.js';
import { createShow } from '../src/show.js';

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf(`--${k}`); return i >= 0 ? argv[i + 1] : d; };
const flag = (k) => argv.includes(`--${k}`);
const even = (v) => Math.max(2, Math.round(v / 2) * 2);

const mode = arg('mode', 'standalone');
const scale = Number(arg('scale', '1'));
const fps = Number(arg('fps', '30'));
const crf = arg('crf', '16');
const preset = arg('preset', 'slow');
const outDir = arg('out', 'out/video');
const audio = arg('audio', 'out/audio/soundtrack.wav');

function encoderArgs(W, H, file) {
  return ['-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${W}x${H}`, '-r', String(fps), '-i', 'pipe:0',
    '-c:v', 'libx264', '-preset', preset, '-tune', 'animation', '-crf', crf, '-pix_fmt', 'yuv420p', '-profile:v', 'high',
    '-x264-params', `rc-lookahead=12:ref=3:bframes=3:keyint=${fps * 5}:min-keyint=${fps}:scenecut=0`,
    '-threads', '2', file];
}

async function worker() {
  const screen = arg('screen');
  const from = Number(arg('from')), to = Number(arg('to'));
  const L = layoutFor(mode, screen);
  const view = L.views.find((v) => v.name === screen) || L.views[0];
  const W = even(SCREENS[screen].w * scale), H = even(SCREENS[screen].h * scale);
  const show = createShow({ createCanvas, Path2D }, L, view, W, H);
  const ff = spawn('ffmpeg', encoderArgs(W, H, arg('seg')), { stdio: ['pipe', 'inherit', 'inherit'] });
  const t0 = Date.now();
  for (let i = from; i < to; i++) {
    const c = show.render(i / fps);
    await new Promise((res, rej) => ff.stdin.write(c.data(), (e) => (e ? rej(e) : res())));
    if ((i - from) % 15 === 14 || i === to - 1) process.stderr.write(`  [${screen} ${from}-${to}] frame ${i + 1 - from}/${to - from}  ${(((Date.now() - t0) / (i + 1 - from)) / 1000).toFixed(2)} s/frame\n`);
  }
  ff.stdin.end();
  const [code] = await once(ff, 'close');
  if (code !== 0) throw new Error(`encoder exited with ${code}`);
}

async function master() {
  const screens = arg('screens', 'left,center,right').split(',');
  const workers = Number(arg('workers', '3'));
  const tag = arg('tag', mode === 'spanned' ? 'SPANNED' : 'STANDALONE');
  mkdirSync(outDir, { recursive: true });
  const total = LOOP * fps;
  const results = [];
  for (const screen of screens) {
    const W = even(SCREENS[screen].w * scale), H = even(SCREENS[screen].h * scale);
    const segDir = `${outDir}/_seg_${tag}_${screen}`;
    rmSync(segDir, { recursive: true, force: true });
    mkdirSync(segDir, { recursive: true });
    // big frames need more memory per x264 encoder: allow e.g. --workers-center 2
    const n = Math.min(Number(arg(`workers-${screen}`, workers)), total);
    const bounds = Array.from({ length: n + 1 }, (_, i) => Math.round((i * total) / n));
    const t0 = Date.now();
    console.log(`▶ ${tag} ${screen}: ${W}x${H} @ ${fps} fps, ${total} frames, ${n} workers`);
    await Promise.all(bounds.slice(0, -1).map((a, i) => new Promise((res, rej) => {
      const p = spawn(process.execPath, [fileURLToPath(import.meta.url), '--worker', '--mode', mode, '--screen', screen, '--scale', String(scale), '--fps', String(fps), '--crf', crf, '--preset', preset, '--from', String(a), '--to', String(bounds[i + 1]), '--seg', `${segDir}/${String(i).padStart(2, '0')}.mp4`], { stdio: ['ignore', 'inherit', 'inherit'] });
      p.on('close', (c) => (c === 0 ? res() : rej(new Error(`worker ${i} failed (${c})`))));
    })));
    const list = bounds.slice(0, -1).map((_, i) => `file '${String(i).padStart(2, '0')}.mp4'`).join('\n');
    writeFileSync(`${segDir}/list.txt`, list + '\n');
    const name = `VANKING_MoleShow_15s_${tag}_${screen}_${W}x${H}${scale === 1 ? '' : '_preview'}.mp4`;
    const out = `${outDir}/${name}`;
    const muxArgs = ['-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', `${segDir}/list.txt`];
    if (!flag('no-audio') && existsSync(audio)) muxArgs.push('-i', audio, '-map', '0:v', '-map', '1:a', '-c:a', 'aac', '-b:a', '320k', '-ar', '48000');
    muxArgs.push('-c:v', 'copy', '-t', String(LOOP), '-movflags', '+faststart', out);
    await new Promise((res, rej) => spawn('ffmpeg', muxArgs, { stdio: 'inherit' }).on('close', (c) => (c === 0 ? res() : rej(new Error('mux failed')))));
    rmSync(segDir, { recursive: true, force: true });
    const mb = (statSync(out).size / 1048576).toFixed(1);
    console.log(`✔ ${out} (${mb} MB) in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
    results.push(out);
  }
  return results;
}

if (flag('worker')) await worker();
else await master();
