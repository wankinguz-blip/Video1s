// Exports the score and the VANKING particle targets to JSON for the Blender pipeline, so the
// render machine needs only Blender + ffmpeg (no Node).  Run: node scripts/export-blender-data.mjs
import { writeFileSync, mkdirSync } from 'node:fs';
import { CHORDS, BURST_BEATS, LAUNCH_BEATS, DRUMS, BASS, SAX, VOCAL, SYNTH } from '../src/music/score.js';
import { TEXT_POINTS, TEXT_WIDTH } from '../src/assets/vanking-text.js';
import { LOOP, BPM, BEAT, LOOP_BEATS, FPS } from '../src/config.js';

mkdirSync('blender/data', { recursive: true });
const score = { loop: LOOP, bpm: BPM, beat: BEAT, loopBeats: LOOP_BEATS, fps: FPS, chords: CHORDS, burstBeats: BURST_BEATS, launchBeats: LAUNCH_BEATS, drums: DRUMS, bass: BASS, sax: SAX, vocal: VOCAL, synth: SYNTH };
writeFileSync('blender/data/score.json', JSON.stringify(score, null, 1) + '\n');
const r = (v) => Math.round(v * 1e4) / 1e4;
const text = { word: 'VANKING', widthInCapHeights: TEXT_WIDTH, columns: ['x', 'y', 'letter', 'edge'], points: TEXT_POINTS.map(([x, y, l, e]) => [r(x), r(y), l, e ? 1 : 0]) };
writeFileSync('blender/data/vanking_text.json', JSON.stringify(text) + '\n');
console.log(`score.json: ${SAX.length} sax notes, ${VOCAL.length} syllables; vanking_text.json: ${TEXT_POINTS.length} points`);
