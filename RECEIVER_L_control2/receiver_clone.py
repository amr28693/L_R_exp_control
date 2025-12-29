#!/usr/bin/env python3
"""
H.E.R.B.I.E. - J-SERIES (J16Cx): Jam View
========================================
Same NLSE dynamics as D16C, but optimized for live jamming:

J16Cx CHANGES (performance layer ONLY):
1) Separate persistence database (new SAVE_DIR + files)
2 Visuals:
   - long-exposure "trail" intensity view
   - minimal HUD (toggleable)
3) Live controls (keypress in the matplotlib window):
   - [ / ] : transpose key down/up (root_midi)
   - 1..4  : set max voices
   - k     : cycle scales (pentatonic / dorian / chromatic / major pentatonic)
   - h     : toggle HUD (minimal/full)
   - d     : toggle dream enable/disable (locks out dreaming when off)
   - p     : cycle palettes

Press Ctrl+C to exit (state saved).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sounddevice as sd
import time
import os
import json
from collections import deque
from datetime import datetime
import hashlib

np.seterr(divide='ignore', invalid='ignore')

# =============================================================================
# PERSISTENCE - J-SERIES (local load)
# =============================================================================
J_SERIES = "J16Cx"
SAVE_DIR = os.getcwd()

STATE_FILE      = os.path.join(SAVE_DIR, "state_J16Cx_MOTHER.npz")
HISTORY_FILE    = os.path.join(SAVE_DIR, "history_J16Cx_MOTHER.json")
ATTRACTORS_FILE = os.path.join(SAVE_DIR, "attractors_J16Cx_MOTHER.json")
PATHS_FILE      = os.path.join(SAVE_DIR, "paths_J16Cx_MOTHER.json")

def ensure_save_dir():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"[HERBIE {J_SERIES}] Created home directory: {SAVE_DIR}")

# =============================================================================
# PARAMETERS (UNCHANGED CORE)
# =============================================================================
L = 14
Nx, Ny = 128, 128
dx, dy = L / Nx, L / Ny
x = np.linspace(-L/2, L/2, Nx)
y = np.linspace(-L/2, L/2, Ny)
X, Y = np.meshgrid(x, y)

dt = 0.015

kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing="xy")
k2 = KX**2 + KY**2
L_op = np.exp(-1j * k2 * dt / 2)

SAMPLE_RATE = 44100
BLOCK_SIZE = 2048

# Regime transition parameters
SILENCE_THRESHOLD_DREAM = 100
SILENCE_FULL_DREAM = 300
DREAM_G_SCALE = 0.6
G_SMOOTHING = 0.95

# =============================================================================
# MUSIC HELPERS (J-SERIES)
# =============================================================================
def hz_to_midi(hz):
    return 69 + 12 * np.log2((hz + 1e-12) / 440.0)

def midi_to_hz(m):
    return 440.0 * (2 ** ((m - 69) / 12))

def quantize_hz_to_scale(hz, root_midi=48, scale=(0, 3, 5, 7, 10), lo=36, hi=84):
    """
    Quantize frequency to nearest pitch in a scale.
    - root_midi: key center (48 = C2)
    - scale: semitone offsets in octave
    """
    if hz <= 1e-6:
        return 0.0
    m = float(np.clip(hz_to_midi(hz), lo, hi))

    best = None
    best_err = 1e9
    # search nearby octaves
    for octv in range(-6, 7):
        for deg in scale:
            cand = root_midi + deg + 12 * octv
            err = abs(cand - m)
            if err < best_err:
                best_err = err
                best = cand
    return midi_to_hz(best)

# Some nice live sets
SCALES = [
    ("minor_pent", (0, 3, 5, 7, 10)),
    ("dorian", (0, 2, 3, 5, 7, 9, 10)),
    ("chromatic", tuple(range(12))),
    ("major_pent", (0, 2, 4, 7, 9)),
]

# =============================================================================
# PHASE VISUALIZATION & VORTEX DETECTION
# =============================================================================
def complex_to_rgb(psi):
    amplitude = np.abs(psi)
    phase = np.angle(psi)

    amp_norm = amplitude / (np.max(amplitude) + 1e-10)
    hue = (phase + np.pi) / (2 * np.pi)

    hsv = np.zeros((Ny, Nx, 3))
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = 0.9
    hsv[:, :, 2] = amp_norm

    rgb = mcolors.hsv_to_rgb(hsv)
    return rgb

def count_vortices(psi):
    phase = np.angle(psi)

    def phase_diff(p1, p2):
        diff = p2 - p1
        return np.arctan2(np.sin(diff), np.cos(diff))

    p_tl = phase[:-1, :-1]
    p_tr = phase[:-1, 1:]
    p_br = phase[1:, 1:]
    p_bl = phase[1:, :-1]

    d1 = phase_diff(p_tl, p_tr)
    d2 = phase_diff(p_tr, p_br)
    d3 = phase_diff(p_br, p_bl)
    d4 = phase_diff(p_bl, p_tl)

    winding = (d1 + d2 + d3 + d4) / (2 * np.pi)

    positive = np.sum(winding > 0.5)
    negative = np.sum(winding < -0.5)

    return int(positive + negative), int(positive), int(negative)

# =============================================================================
# PATH MEMORY - With surprise tracking (UNCHANGED)
# =============================================================================
class PathMemory:
    def __init__(self):
        self.transitions = {}
        self.waking_transitions = {}
        self.dream_transitions = {}
        self.recent_fps = deque(maxlen=20)
        self.known_paths = {}
        self.path_length = 5
        self.last_fp = None
        self.path_recognition = 0.0
        self.transition_recognition = 0.0
        self.anticipated_fp = None
        self.anticipation_confidence = 0.0
        self.total_transitions = 0
        self.total_paths = 0
        self.dream_paths = 0
        self.waking_paths = 0

        self.last_surprise = 0.0
        self.surprise_accumulator = 0.0
        self.prediction_was_correct = True

    def record_transition(self, from_fp, to_fp, is_dreaming=False):
        if from_fp is None or to_fp is None:
            return
        if from_fp == to_fp:
            return

        if self.anticipated_fp is not None and self.anticipation_confidence > 0.3:
            if to_fp != self.anticipated_fp:
                self.last_surprise = self.anticipation_confidence
                self.surprise_accumulator = min(1.0, self.surprise_accumulator + self.last_surprise * 0.3)
                self.prediction_was_correct = False
            else:
                self.last_surprise = 0.0
                self.prediction_was_correct = True
                self.surprise_accumulator *= 0.95
        else:
            self.last_surprise = 0.0
            self.prediction_was_correct = True

        key = (from_fp, to_fp)

        self.transitions[key] = self.transitions.get(key, 0) + 1
        self.total_transitions += 1

        if is_dreaming:
            self.dream_transitions[key] = self.dream_transitions.get(key, 0) + 1
        else:
            self.waking_transitions[key] = self.waking_transitions.get(key, 0) + 1

        count = self.transitions[key]
        self.transition_recognition = min(1.0, count / 50)

    def update_sequence(self, fp, mood, timestamp, is_dreaming=False):
        if fp is None:
            return

        if self.last_fp is not None and self.last_fp != fp:
            self.record_transition(self.last_fp, fp, is_dreaming)

        self.recent_fps.append(fp)
        self.last_fp = fp

        if len(self.recent_fps) >= self.path_length:
            self._check_path_recognition(mood, timestamp, is_dreaming)
            self._update_anticipation()

    def _hash_path(self, fp_sequence):
        combined = '|'.join(fp_sequence)
        return hashlib.md5(combined.encode()).hexdigest()[:10]

    def _check_path_recognition(self, mood, timestamp, is_dreaming):
        recent = list(self.recent_fps)[-self.path_length:]
        if len(recent) < self.path_length:
            return

        path_hash = self._hash_path(recent)

        if path_hash in self.known_paths:
            info = self.known_paths[path_hash]
            info['count'] += 1
            info['last_seen'] = timestamp
            info['avg_mood'] = 0.9 * info['avg_mood'] + 0.1 * mood
            self.path_recognition = min(1.0, info['count'] / 30)
        else:
            regime = 'dream' if is_dreaming else 'waking'
            self.known_paths[path_hash] = {
                'count': 1,
                'first_seen': timestamp,
                'last_seen': timestamp,
                'avg_mood': float(mood),
                'regime': regime,
                'sequence': recent.copy()
            }
            self.total_paths += 1
            if is_dreaming:
                self.dream_paths += 1
            else:
                self.waking_paths += 1
            self.path_recognition = 0.0

    def _update_anticipation(self):
        if len(self.recent_fps) < 2:
            self.anticipated_fp = None
            self.anticipation_confidence = 0.0
            return

        current_fp = self.recent_fps[-1]
        candidates = {}
        for (from_fp, to_fp), count in self.transitions.items():
            if from_fp == current_fp:
                candidates[to_fp] = candidates.get(to_fp, 0) + count

        if not candidates:
            self.anticipated_fp = None
            self.anticipation_confidence = 0.0
            return

        total = sum(candidates.values())
        best_fp = max(candidates, key=candidates.get)
        self.anticipated_fp = best_fp
        self.anticipation_confidence = candidates[best_fp] / total

    def get_momentum(self):
        return 0.6 * self.path_recognition + 0.4 * self.anticipation_confidence

    def get_surprise(self):
        return float(self.last_surprise)

    def get_accumulated_surprise(self):
        return float(self.surprise_accumulator)

    def get_stats(self):
        return {
            'num_transitions': len(self.transitions),
            'total_transition_count': self.total_transitions,
            'num_paths': len(self.known_paths),
            'waking_paths': self.waking_paths,
            'dream_paths': self.dream_paths,
            'path_recognition': float(self.path_recognition),
            'transition_recognition': float(self.transition_recognition),
            'anticipation_confidence': float(self.anticipation_confidence),
            'momentum': float(self.get_momentum()),
            'surprise': float(self.last_surprise),
            'accumulated_surprise': float(self.surprise_accumulator)
        }

    def save(self, filepath):
        transitions_serializable = {f"{k[0]}|{k[1]}": v for k, v in self.transitions.items()}
        waking_serializable = {f"{k[0]}|{k[1]}": v for k, v in self.waking_transitions.items()}
        dream_serializable = {f"{k[0]}|{k[1]}": v for k, v in self.dream_transitions.items()}

        data = {
            'transitions': transitions_serializable,
            'waking_transitions': waking_serializable,
            'dream_transitions': dream_serializable,
            'known_paths': self.known_paths,
            'total_transitions': self.total_transitions,
            'total_paths': self.total_paths,
            'waking_paths': self.waking_paths,
            'dream_paths': self.dream_paths
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            def deserialize_transitions(raw):
                result = {}
                for k, v in raw.items():
                    parts = k.split('|')
                    if len(parts) == 2:
                        result[(parts[0], parts[1])] = v
                return result

            self.transitions = deserialize_transitions(data.get('transitions', {}))
            self.waking_transitions = deserialize_transitions(data.get('waking_transitions', {}))
            self.dream_transitions = deserialize_transitions(data.get('dream_transitions', {}))
            self.known_paths = data.get('known_paths', {})
            self.total_transitions = int(data.get('total_transitions', 0))
            self.total_paths = int(data.get('total_paths', 0))
            self.waking_paths = int(data.get('waking_paths', 0))
            self.dream_paths = int(data.get('dream_paths', 0))
            return True
        except:
            return False

# =============================================================================
# ATTRACTOR MEMORY - With novelty signaling (UNCHANGED)
# =============================================================================
class AttractorMemory:
    def __init__(self):
        self.known_attractors = {}
        self.current_fingerprint = None
        self.current_coarse = None
        self.recognition_strength = 0.0
        self.is_stable = False
        self.stability_frames = 0
        self.total_visits = 0
        self.dream_attractors = 0
        self.waking_attractors = 0

        self.just_discovered_new = False
        self.novelty_signal = 0.0

    def fingerprint_field(self, psi):
        I = np.abs(psi)**2

        block_size = Nx // 8
        coarse = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                block = I[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                coarse[i, j] = np.mean(block)

        coarse = coarse / (np.max(coarse) + 1e-10)
        quantized = np.digitize(coarse, [0.25, 0.5, 0.75])
        fingerprint = hashlib.md5(quantized.tobytes()).hexdigest()[:8]

        return fingerprint, coarse

    def update(self, psi, mood, timestamp, is_dreaming=False):
        new_fp, coarse = self.fingerprint_field(psi)
        self.current_coarse = coarse

        if new_fp == self.current_fingerprint:
            self.stability_frames += 1
        else:
            self.stability_frames = max(0, self.stability_frames - 5)

        self.current_fingerprint = new_fp
        self.is_stable = self.stability_frames > 10

        self.just_discovered_new = False
        self.novelty_signal *= 0.9

        if self.is_stable:
            if new_fp in self.known_attractors:
                info = self.known_attractors[new_fp]
                info['count'] += 1
                info['last_seen'] = timestamp
                info['avg_mood'] = 0.9 * info['avg_mood'] + 0.1 * mood
                self.recognition_strength = min(1.0, info['count'] / 100)
            else:
                regime = 'dream' if is_dreaming else 'waking'
                self.known_attractors[new_fp] = {
                    'count': 1,
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'avg_mood': float(mood),
                    'regime': regime
                }
                self.recognition_strength = 0.0
                self.just_discovered_new = True
                self.novelty_signal = 1.0

                if is_dreaming:
                    self.dream_attractors += 1
                else:
                    self.waking_attractors += 1
            self.total_visits += 1
        else:
            self.recognition_strength *= 0.95

        return new_fp

    def get_confidence(self):
        stability_factor = min(1.0, self.stability_frames / 20)
        return float(self.recognition_strength * stability_factor)

    def get_novelty(self):
        return float(self.novelty_signal)

    def get_stats(self):
        return {
            'num_known': len(self.known_attractors),
            'total_visits': int(self.total_visits),
            'waking_attractors': int(self.waking_attractors),
            'dream_attractors': int(self.dream_attractors),
            'recognition_strength': float(self.recognition_strength),
            'is_stable': bool(self.is_stable),
            'stability_frames': int(self.stability_frames),
            'current_fp': self.current_fingerprint,
            'confidence': float(self.get_confidence()),
            'novelty': float(self.novelty_signal),
            'just_discovered': bool(self.just_discovered_new)
        }

    def save(self, filepath):
        data = {
            'known_attractors': self.known_attractors,
            'total_visits': self.total_visits,
            'waking_attractors': self.waking_attractors,
            'dream_attractors': self.dream_attractors
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.known_attractors = data.get('known_attractors', {})
            self.total_visits = int(data.get('total_visits', 0))
            self.waking_attractors = int(data.get('waking_attractors', 0))
            self.dream_attractors = int(data.get('dream_attractors', 0))
            return True
        except:
            return False

# =============================================================================
# MULTI-SCALE AUDIO (UNCHANGED)
# =============================================================================
class MultiScaleAudio:
    def __init__(self):
        self.amplitude = 0.0
        self.pitch_hz = 0.0
        self.centroid = 0.0
        self.flux = 0.0

        self.amp_short = deque(maxlen=150)
        self.pitch_short = deque(maxlen=150)
        self.centroid_short = deque(maxlen=150)

        self.amp_long = deque(maxlen=900)
        self.pitch_long = deque(maxlen=900)
        self.centroid_long = deque(maxlen=900)

        self.amp_short_avg = 0.0
        self.amp_long_avg = 0.0
        self.pitch_short_avg = 0.0
        self.pitch_long_avg = 0.0
        self.centroid_short_avg = 0.0
        self.centroid_long_avg = 0.0

        self.amp_trend = 0.0
        self.complexity = 0.0

        self.buffer = np.zeros(4096)
        self.prev_spectrum = np.zeros(2049)
        self.silence_frames = 0

        self.amp_hist = deque(maxlen=300)
        self.pitch_hist = deque(maxlen=300)

    def process(self, indata):
        if len(indata.shape) > 1:
            audio = indata[:, 0]
        else:
            audio = indata

        self.amplitude = float(np.sqrt(np.mean(audio**2)))

        self.buffer = np.roll(self.buffer, -len(audio))
        self.buffer[-len(audio):] = audio

        if self.amplitude > 0.01:
            self.pitch_hz = float(self._detect_pitch(self.buffer))
        else:
            self.pitch_hz *= 0.9

        spectrum = np.abs(np.fft.rfft(self.buffer * np.hanning(len(self.buffer))))
        freqs = np.fft.rfftfreq(len(self.buffer), 1/SAMPLE_RATE)

        if np.sum(spectrum) > 1e-10:
            self.centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))

        self.flux = float(np.sqrt(np.mean((spectrum - self.prev_spectrum)**2)))
        self.prev_spectrum = spectrum.copy()

        self.amp_short.append(self.amplitude)
        self.amp_long.append(self.amplitude)
        self.pitch_short.append(self.pitch_hz)
        self.pitch_long.append(self.pitch_hz)
        self.centroid_short.append(self.centroid)
        self.centroid_long.append(self.centroid)

        if len(self.amp_short) > 10:
            self.amp_short_avg = float(np.mean(self.amp_short))
            self.pitch_short_avg = float(np.mean([p for p in self.pitch_short if p > 50]) or 0)
            self.centroid_short_avg = float(np.mean(self.centroid_short))

            self.complexity = float(np.std(self.amp_short) / (self.amp_short_avg + 1e-10))

            recent = list(self.amp_short)[-30:]
            older = list(self.amp_short)[:30]
            if len(recent) > 0 and len(older) > 0:
                self.amp_trend = float(np.mean(recent) - np.mean(older))

        if len(self.amp_long) > 100:
            self.amp_long_avg = float(np.mean(self.amp_long))
            self.pitch_long_avg = float(np.mean([p for p in self.pitch_long if p > 50]) or 0)
            self.centroid_long_avg = float(np.mean(self.centroid_long))

        if self.amplitude < 0.008:
            self.silence_frames += 1
        else:
            self.silence_frames = 0

        self.amp_hist.append(self.amplitude)
        self.pitch_hist.append(self.pitch_hz)

    def _detect_pitch(self, audio):
        audio = audio - np.mean(audio)
        if np.max(np.abs(audio)) < 0.001:
            return 0.0
        audio = audio / (np.max(np.abs(audio)) + 1e-10)

        n = len(audio)
        fft = np.fft.fft(audio, n=2*n)
        autocorr = np.fft.ifft(fft * np.conj(fft))[:n].real
        autocorr = autocorr / (autocorr[0] + 1e-10)

        min_lag = int(SAMPLE_RATE / 1000)
        max_lag = int(SAMPLE_RATE / 60)

        if max_lag <= min_lag:
            return 0.0

        search = autocorr[min_lag:max_lag]

        peaks = []
        for i in range(1, len(search) - 1):
            if search[i] > search[i-1] and search[i] > search[i+1] and search[i] > 0.2:
                peaks.append((i + min_lag, search[i]))

        if not peaks:
            return 0.0

        best_lag = max(peaks, key=lambda x: x[1])[0]
        return float(SAMPLE_RATE / best_lag)

    def get_normalized_features(self):
        rel_amp = self.amplitude / (self.amp_long_avg + 1e-10) if self.amp_long_avg > 0.01 else 0
        rel_amp = float(np.clip(rel_amp, 0, 3))

        rel_centroid = self.centroid / (self.centroid_long_avg + 1e-10) if self.centroid_long_avg > 100 else 0
        rel_centroid = float(np.clip(rel_centroid, 0, 3))

        return {
            'rel_amp': rel_amp,
            'rel_centroid': rel_centroid,
            'complexity': float(self.complexity),
            'trend': float(self.amp_trend),
            'pitch': float(self.pitch_hz)
        }

# =============================================================================
# FIELD SYNTH (J-SERIES: quantized + smoother + PA-safe)
# =============================================================================
class FieldSynth:
    def __init__(self):
        self.phases = np.zeros(8)
        self.frequencies = [220.0]
        self.amplitudes = [0.0]
        self.output_amp = 0.0
        self.primary_freq = 220.0
        self.num_voices = 1

        # Jam controls
        self.amp_smooth = 0.0
        self.root_midi = 48  # C2
        self.scale_name, self.scale = SCALES[0]
        self.max_voices = 3
        self.limiter_drive = 6000000.5

    def set_scale(self, name, scale):
        self.scale_name = name
        self.scale = scale

    def update_from_field(self, psi, field_energy, input_amp, mood, confidence, momentum, dream_depth):
        I = np.abs(psi)**2
        threshold = np.max(I) * 0.25
        peaks = []

        for i in range(2, Ny-2):
            for j in range(2, Nx-2):
                if I[i, j] > threshold:
                    if (I[i, j] > I[i-1, j] and I[i, j] > I[i+1, j] and
                        I[i, j] > I[i, j-1] and I[i, j] > I[i, j+1]):
                        px = (j - Nx/2) / (Nx/2)
                        py = (i - Ny/2) / (Ny/2)
                        intensity = I[i, j]
                        local_phase = np.angle(psi[i, j])
                        peaks.append({
                            'r': np.sqrt(px**2 + py**2),
                            'intensity': intensity,
                            'phase': local_phase
                        })

        peaks = sorted(peaks, key=lambda p: p['intensity'], reverse=True)[:self.max_voices]

        if not peaks:
            base_freq = 200.0 - dream_depth * 60
            base_freq = quantize_hz_to_scale(base_freq, root_midi=self.root_midi, scale=self.scale)
            self.frequencies = [base_freq]
            self.amplitudes = [0.03]
            self.num_voices = 1
        else:
            self.frequencies = []
            self.amplitudes = []

            for p in peaks:
                freq = 80 + p['r'] * 720
                freq *= (1 + 0.1 * np.sin(p['phase']))
                freq *= (1 + 0.1 * momentum)
                freq *= (1.0 - 0.3 * dream_depth)

                # JAM: snap to scale
                freq = quantize_hz_to_scale(freq, root_midi=self.root_midi, scale=self.scale, lo=36, hi=84)

                amp = np.sqrt(p['intensity']) * 0.18
                self.frequencies.append(freq)
                self.amplitudes.append(float(amp))

            self.num_voices = len(peaks)

        mood_factor = 0.5 + mood * 0.5
        confidence_factor = 0.8 + confidence * 0.4
        momentum_factor = 0.9 + momentum * 0.2
        dream_factor = 1.0 - 0.5 * dream_depth

        input_gate = np.clip(1.0 - input_amp * 5, 0.2, 1.0)
        energy_norm = np.clip(field_energy / 50, 0.1, 1.0)

        target_amp = (energy_norm * input_gate * 0.3 *
                      mood_factor * confidence_factor * momentum_factor * dream_factor)

        # Smooth output amplitude (jam stability)
        self.amp_smooth = 0.96 * self.amp_smooth + 0.04 * target_amp
        self.output_amp = float(self.amp_smooth)

        if self.frequencies:
            self.primary_freq = float(self.frequencies[0])

    def generate(self, num_samples):
        t = np.arange(num_samples) / SAMPLE_RATE
        output = np.zeros(num_samples, dtype=np.float64)

        for i, (freq, amp) in enumerate(zip(self.frequencies, self.amplitudes)):
            if i >= len(self.phases):
                continue
            wave = np.sin(2 * np.pi * freq * t + self.phases[i])
            output += wave * amp * self.output_amp
            self.phases[i] += 2 * np.pi * freq * num_samples / SAMPLE_RATE
            self.phases[i] = self.phases[i] % (2 * np.pi)

        # PA safety: soft limiter / clip
        output = np.tanh(output * self.limiter_drive)
        output *= 0.35
        return np.column_stack([output, output]).astype(np.float32)

# =============================================================================
# HERBIE (same brain, J-series persistence + optional dream lock)
# =============================================================================
class HERBIE:
    def __init__(self):
        ensure_save_dir()
        self.log_file = os.path.join(SAVE_DIR, f"RECEIVERprosecutor_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")


        loaded = self._load_state()

        if loaded:
            print(f"[HERBIE {J_SERIES}] Restored from previous session")
            if self.mood > 0.9:
                print(f"[HERBIE {J_SERIES}] Resetting saturated mood from {self.mood:.2f} to 0.5")
                self.mood = 0.5
        else:
            print(f"[HERBIE {J_SERIES}] Starting fresh")
            self.psi = np.exp(-(X**2 + Y**2) / 3).astype(np.complex128)
            self.psi += 0.01 * (np.random.randn(Ny, Nx) + 1j * np.random.randn(Ny, Nx))

            self.mood = 0.5
            self.total_runtime = 0.0
            self.total_interactions = 0
            self.birth_date = datetime.now().isoformat()

        self.audio = MultiScaleAudio()
        self.synth = FieldSynth()

        self.attractor_memory = AttractorMemory()
        self.attractor_memory.load(ATTRACTORS_FILE)

        self.path_memory = PathMemory()
        self.path_memory.load(PATHS_FILE)

        self.dream_depth = 0.0
        self.is_dreaming = False
        self.dream_enabled = True  # J-series live toggle
        self.current_g = 0.0
        self.smoothed_g = 0.0

        self.vortex_count = 0
        self.vortex_positive = 0
        self.vortex_negative = 0

        self.entropy_norm = 0.0
        self.coherence_norm = 0.0
        self.ce_radius = 0.0
        self.ce_theta = 0.0
        self.dtheta = 0.0

        self.coherence = 0.0
        self.entropy = 0.0
        self.field_energy = 0.0

        self.entropy_min = 1.0
        self.entropy_max = 10.0
        self.coherence_min = 1.0
        self.coherence_max = 50.0

        self.session_start = time.time()
        self.session_interactions = 0
        self.step_count = 0

        self.surprise = 0.0
        self.novelty = 0.0
        self.boredom = 0.0

        self.mood_hist = deque(maxlen=300)
        self.confidence_hist = deque(maxlen=300)
        self.momentum_hist = deque(maxlen=300)
        self.dream_hist = deque(maxlen=300)
        self.g_hist = deque(maxlen=300)
        self.vortex_hist = deque(maxlen=300)
        self.surprise_hist = deque(maxlen=300)
        self.boredom_hist = deque(maxlen=300)
        self.theta_hist = deque(maxlen=300)
        self.dtheta_hist = deque(maxlen=300)

        self.ce_trajectory = deque(maxlen=200)

        self.in_stream = None
        self.out_stream = None

    def _load_state(self):
        try:
            if os.path.exists(STATE_FILE):
                data = np.load(STATE_FILE, allow_pickle=True)
                self.psi = data['psi']
                self.mood = float(data['mood'])
                self.total_runtime = float(data['total_runtime'])
                self.total_interactions = int(data['total_interactions'])
                self.birth_date = str(data['birth_date'])
                return True
        except Exception as e:
            print(f"[HERBIE {J_SERIES}] Could not load state: {e}")
        return False

    def save_state(self):
        try:
            self.total_runtime += time.time() - self.session_start
            self.total_interactions += self.session_interactions

            np.savez(
                STATE_FILE,
                psi=self.psi,
                mood=self.mood,
                total_runtime=self.total_runtime,
                total_interactions=self.total_interactions,
                birth_date=self.birth_date
            )

            self.attractor_memory.save(ATTRACTORS_FILE)
            self.path_memory.save(PATHS_FILE)

            attr_stats = self.attractor_memory.get_stats()
            path_stats = self.path_memory.get_stats()

            history = {
                'series': J_SERIES,
                'birth_date': self.birth_date,
                'total_runtime_hours': self.total_runtime / 3600,
                'total_interactions': self.total_interactions,
                'last_session': datetime.now().isoformat(),
                'last_mood': float(self.mood),
                'dream_enabled': bool(self.dream_enabled),
                'synth': {
                    'root_midi': int(self.synth.root_midi),
                    'scale': self.synth.scale_name,
                    'max_voices': int(self.synth.max_voices),
                    'limiter_drive': float(self.synth.limiter_drive),
                },
                'attractors': {
                    'total': attr_stats['num_known'],
                    'waking': attr_stats['waking_attractors'],
                    'dream': attr_stats['dream_attractors']
                },
                'paths': {
                    'total': path_stats['num_paths'],
                    'waking': path_stats['waking_paths'],
                    'dream': path_stats['dream_paths']
                },
                'transitions': path_stats['total_transition_count']
            }
            with open(HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)

            print(f"[HERBIE {J_SERIES}] State saved. Lifetime: {self.total_runtime/3600:.2f}h")
            print(f"                  Attractors: {attr_stats['num_known']} "
                  f"(W:{attr_stats['waking_attractors']} D:{attr_stats['dream_attractors']})")
            print(f"                  Paths: {path_stats['num_paths']} "
                  f"(W:{path_stats['waking_paths']} D:{path_stats['dream_paths']})")
            return True
        except Exception as e:
            print(f"[HERBIE {J_SERIES}] Could not save state: {e}")
            return False

    def start(self):
        self.in_stream = sd.InputStream(
            callback=self._in_cb,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE
        )
        self.in_stream.start()

        self.out_stream = sd.OutputStream(
            callback=self._out_cb,
            channels=2,
            samplerate=SAMPLE_RATE,
            blocksize=1024
        )
        self.out_stream.start()

        age = self.total_runtime / 3600
        attr_stats = self.attractor_memory.get_stats()
        path_stats = self.path_memory.get_stats()

        print(f"[HERBIE {J_SERIES}] Awake. Age: {age:.1f}h, Mood: {self.mood:.2f}")
        print(f"                  Synth: root_midi={self.synth.root_midi} scale={self.synth.scale_name} voices={self.synth.max_voices}")
        print(f"                  Attractors: {attr_stats['num_known']} "
              f"(W:{attr_stats['waking_attractors']} D:{attr_stats['dream_attractors']})")
        print(f"                  Paths: {path_stats['num_paths']} "
              f"(W:{path_stats['waking_paths']} D:{path_stats['dream_paths']})")
        print("                  Controls: [ ] transpose | 1..4 voices | s scale | d dream | h HUD | p palette")

    def stop(self):
        if self.in_stream:
            self.in_stream.stop()
            self.in_stream.close()
        if self.out_stream:
            self.out_stream.stop()
            self.out_stream.close()

    def _in_cb(self, indata, frames, time_info, status):
        self.audio.process(indata)

    def _out_cb(self, outdata, frames, time_info, status):
        outdata[:] = self.synth.generate(frames)

    def _update_mood(self, confidence, momentum, surprise, novelty):
        self.boredom = confidence * momentum * (1.0 - surprise) * (1.0 - novelty)

        mood_delta = 0.0

        if surprise > 0.1:
            headroom = max(0.0, 0.85 - self.mood)
            surprise_boost = surprise * 0.1 * (0.5 + headroom)
            mood_delta += surprise_boost

        if novelty > 0.5:
            headroom = max(0.0, 0.85 - self.mood)
            novelty_boost = 0.04 * (0.5 + headroom)
            mood_delta += novelty_boost

        if self.boredom > 0.3:
            boredom_target = 0.4
            boredom_pull = (self.boredom - 0.3) * 0.03
            mood_delta += (boredom_target - self.mood) * boredom_pull

        if self.mood > 0.7 and surprise < 0.1 and novelty < 0.3:
            ceiling_decay = (self.mood - 0.7) * 0.008
            mood_delta -= ceiling_decay

        if self.audio.amplitude < 0.01:
            silence_target = 0.3
            mood_delta += (silence_target - self.mood) * 0.002

        if self.is_dreaming:
            dream_target = 0.4
            mood_delta += (dream_target - self.mood) * 0.005 * self.dream_depth

        if self.audio.amplitude > 0.02:
            self.session_interactions += 1

        self.mood = float(np.clip(self.mood + mood_delta, 0.15, 0.9))

    def compute_metrics(self):
        I = np.abs(self.psi)**2
        I_sum = np.sum(I) + 1e-12
        I_norm = I / I_sum

        self.coherence = float(np.sum(I_norm**2) * Nx * Ny)
        self.field_energy = float(I_sum)

        I_flat = I_norm.flatten()
        I_flat = I_flat[I_flat > 1e-12]
        self.entropy = float(-np.sum(I_flat * np.log(I_flat)))

        self.entropy_min = float(min(self.entropy_min, self.entropy))
        self.entropy_max = float(max(self.entropy_max, self.entropy))
        self.coherence_min = float(min(self.coherence_min, self.coherence))
        self.coherence_max = float(max(self.coherence_max, self.coherence))

        h_range = self.entropy_max - self.entropy_min + 1e-10
        c_range = self.coherence_max - self.coherence_min + 1e-10

        self.entropy_norm = float((self.entropy - self.entropy_min) / h_range)
        self.coherence_norm = float((self.coherence - self.coherence_min) / c_range)

        self.ce_radius = float(np.sqrt(self.entropy_norm**2 + self.coherence_norm**2))

        old_theta = self.ce_theta
        self.ce_theta = float(np.arctan2(self.coherence_norm + 1e-6, self.entropy_norm + 1e-6))

        if len(self.theta_hist) > 0:
            dtheta = self.ce_theta - old_theta
            if dtheta > np.pi:
                dtheta -= 2 * np.pi
            elif dtheta < -np.pi:
                dtheta += 2 * np.pi
            self.dtheta = float(dtheta)
        else:
            self.dtheta = 0.0

        self.ce_trajectory.append((self.entropy_norm, self.coherence_norm))

        self.vortex_count, self.vortex_positive, self.vortex_negative = count_vortices(self.psi)

        timestamp = datetime.now().isoformat()

        current_fp = self.attractor_memory.update(self.psi, self.mood, timestamp, self.is_dreaming)
        self.path_memory.update_sequence(current_fp, self.mood, timestamp, self.is_dreaming)

        confidence = self.attractor_memory.get_confidence()
        momentum = self.path_memory.get_momentum()
        self.surprise = float(self.path_memory.get_surprise())
        self.novelty = float(self.attractor_memory.get_novelty())

        self._update_mood(confidence, momentum, self.surprise, self.novelty)

        self.synth.update_from_field(
            self.psi, self.field_energy, self.audio.amplitude,
            self.mood, confidence, momentum, self.dream_depth
        )

        self.mood_hist.append(self.mood)
        self.confidence_hist.append(float(confidence))
        self.momentum_hist.append(float(momentum))
        self.dream_hist.append(float(self.dream_depth))
        self.g_hist.append(float(self.smoothed_g))
        self.vortex_hist.append(int(self.vortex_count))
        self.surprise_hist.append(float(self.surprise))
        self.boredom_hist.append(float(self.boredom))
        self.theta_hist.append(float(self.ce_theta))
        self.dtheta_hist.append(float(self.dtheta))

    def evolve(self):
        features = self.audio.get_normalized_features()
        amp = float(self.audio.amplitude)
        pitch = float(self.audio.pitch_hz)
        flux = float(self.audio.flux)

        rel_amp = float(features['rel_amp'])
        complexity = float(features['complexity'])

        silence = int(self.audio.silence_frames)

        # Dream depth (optionally locked out for live play)
        if not self.dream_enabled:
            self.dream_depth = 0.0
        else:
            if silence < SILENCE_THRESHOLD_DREAM:
                self.dream_depth = max(0.0, self.dream_depth - 0.02)
            else:
                progress = (silence - SILENCE_THRESHOLD_DREAM) / (SILENCE_FULL_DREAM - SILENCE_THRESHOLD_DREAM)
                target_depth = np.clip(progress, 0.0, 1.0)
                self.dream_depth = self.dream_depth * 0.95 + target_depth * 0.05

        self.is_dreaming = bool(self.dream_depth > 0.1)

        # g dynamics (UNCHANGED)
        g_base = (0.3 + 0.4 * self.mood) + 2.0 * rel_amp + complexity

        if self.is_dreaming:
            g_target = -g_base * DREAM_G_SCALE * self.dream_depth
        else:
            g_target = g_base * (1.0 - 0.5 * self.dream_depth)

        self.smoothed_g = float(G_SMOOTHING * self.smoothed_g + (1 - G_SMOOTHING) * g_target)
        self.current_g = self.smoothed_g

        psi_k = np.fft.fft2(self.psi)
        psi_k *= L_op
        psi_lin = np.fft.ifft2(psi_k)

        self.psi = psi_lin * np.exp(1j * self.smoothed_g * dt * np.abs(psi_lin)**2)

        if amp > 0.005 and not self.is_dreaming:
            r = np.sqrt(X**2 + Y**2)
            theta = np.arctan2(Y, X)

            if pitch > 60:
                inject_r = np.clip((pitch - 60) / 500, 0.1, 0.9) * (L/2)
            else:
                rel_cent = float(features['rel_centroid'])
                inject_r = np.clip(rel_cent * 0.4, 0.2, 0.8) * (L/2)

            inject = np.exp(-(r - inject_r)**2 / 0.5)

            if flux > 5:
                inject = inject * np.exp(1j * theta * (1 + flux/20))

            strength = 0.05 * np.sqrt(rel_amp)
            self.psi += strength * inject * np.exp(1j * np.angle(self.psi))

        if self.is_dreaming:
            diss = 0.998
        elif self.audio.silence_frames > 30:
            diss = 0.99
        else:
            diss = 0.999
        self.psi *= diss

        if self.is_dreaming and self.dream_depth > 0.3:
            mood_shift = (self.mood - 0.5) * (L/2)
            phase_drift = (self.total_runtime + time.time() - self.session_start) / 500
            x_offset = mood_shift * np.cos(phase_drift)
            y_offset = mood_shift * np.sin(phase_drift)

            sigma = 2.0 + self.dream_depth * 2.0
            ground = np.exp(-((X - x_offset)**2 + (Y - y_offset)**2) / sigma)

            bias = 0.005 * self.dream_depth
            self.psi = (1-bias) * self.psi + bias * ground * np.exp(1j * np.angle(self.psi))

        noise_scale = 0.0005 * (1 + self.dream_depth)
        self.psi += noise_scale * (np.random.randn(Ny, Nx) + 1j * np.random.randn(Ny, Nx))

        norm = np.sqrt(np.sum(np.abs(self.psi)**2))
        self.psi *= np.clip(25 / norm, 0.5, 2.0)

    def step(self):
        t0_mono = time.perf_counter_ns()
        self.evolve()
        self.compute_metrics()
        self.step_count += 1
        
        # Prosecutor log
        h_id = self.attractor_memory.current_fingerprint
        I = np.abs(self.psi)**2
        q_size = 64
        q_vals = [
            np.mean(I[:q_size, :q_size]),
            np.mean(I[:q_size, q_size:]),
            np.mean(I[q_size:, :q_size]),
            np.mean(I[q_size:, q_size:])
        ]
        log_entry = {
            "mono_ns": t0_mono,
            "step": self.step_count,
            "hash": h_id,
            "features": [float(f"{x:.8f}") for x in q_vals],
            "rms": float(f"{self.audio.amplitude:.6f}")
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

# =============================================================================
# VISUALIZATION (JAM: trails + minimal HUD)
# =============================================================================
class Viz:
    def __init__(self, h: HERBIE):
        self.h = h

        plt.ion()
        self.fig = plt.figure(figsize=(16, 10), facecolor='black')

        # Palette cycling for intensity
        self.palettes = ["inferno", "twilight", "magma", "plasma", "cividis"]
        self.palette_i = 0

        # Intensity w/ trails
        self.ax_intensity = self.fig.add_subplot(251)
        self.ax_intensity.set_facecolor('black')
        self.ax_intensity.set_xticks([]); self.ax_intensity.set_yticks([])
        I = np.abs(h.psi)**2
        self.im_intensity = self.ax_intensity.imshow(
            I, origin='lower', cmap=self.palettes[self.palette_i],
            extent=[-L/2, L/2, -L/2, L/2]
        )
        self.ax_intensity.set_title('INTENSITY (TRAIL)', color='white', fontsize=10)

        self.trail = None
        self.trail_alpha = 0.88  # higher = longer exposure

        # Phase/amp RGB
        self.ax_phase = self.fig.add_subplot(252)
        self.ax_phase.set_facecolor('black')
        self.ax_phase.set_xticks([]); self.ax_phase.set_yticks([])
        rgb = complex_to_rgb(h.psi)
        self.im_phase = self.ax_phase.imshow(rgb, origin='lower',
                                             extent=[-L/2, L/2, -L/2, L/2])
        self.ax_phase.set_title('PHASE + AMP', color='white', fontsize=10)

        # Status panel (minimal by default)
        self.ax_status = self.fig.add_subplot(253)
        self.ax_status.set_facecolor('black')
        self.ax_status.set_xticks([]); self.ax_status.set_yticks([])
        self.ax_status.set_xlim(0, 10); self.ax_status.set_ylim(0, 10)

        self.hud_full = False
        self.txt_name = self.ax_status.text(5, 9.2, f'H.E.R.B.I.E. {J_SERIES}', ha='center',
                                            fontsize=12, color='cyan', fontweight='bold')
        self.txt_age = self.ax_status.text(5, 8.2, 'Age: --', ha='center',
                                           fontsize=7, color='gray')
        self.txt_mode = self.ax_status.text(5, 7.0, 'WAKING', ha='center',
                                            fontsize=11, color='lime')
        self.txt_regime = self.ax_status.text(5, 5.9, 'g = +0.00', ha='center',
                                              fontsize=8, color='orange')
        self.txt_mood = self.ax_status.text(5, 4.8, 'Mood: --', ha='center',
                                            fontsize=8, color='yellow')
        self.txt_music = self.ax_status.text(5, 3.6, 'Key/Scale: --', ha='center',
                                             fontsize=8, color='white')
        self.txt_misc = self.ax_status.text(5, 2.4, '', ha='center',
                                            fontsize=7, color='gray')
        self.txt_help = self.ax_status.text(
            5, 0.7,
            "[ ] key  |  1..4 voices  |  s scale  |  d dream  |  h HUD  |  p palette",
            ha='center', fontsize=6, color='gray', alpha=0.8
        )

        # Mood/surprise/boredom trace (kept)
        self.ax_affect = self.fig.add_subplot(254)
        self.ax_affect.set_facecolor('black')
        self.ax_affect.set_title('MOOD / SURPRISE / BOREDOM', color='yellow', fontsize=9)
        self.ax_affect.tick_params(colors='white', labelsize=7)
        for s in self.ax_affect.spines.values(): s.set_color('gray')
        self.line_mood_trace, = self.ax_affect.plot([], [], color='yellow', lw=1.5, label='mood')
        self.line_surprise_trace, = self.ax_affect.plot([], [], color='red', lw=1, alpha=0.7, label='surprise')
        self.line_boredom_trace, = self.ax_affect.plot([], [], color='gray', lw=1, alpha=0.7, label='boredom')
        self.ax_affect.set_ylim(0, 1)
        self.ax_affect.axhline(0.5, color='white', linestyle=':', alpha=0.2)
        self.ax_affect.legend(loc='upper right', fontsize=6, framealpha=0.3)

        # Theta time series
        self.ax_theta = self.fig.add_subplot(255)
        self.ax_theta.set_facecolor('black')
        self.ax_theta.set_title('θ REGIME', color='cyan', fontsize=9)
        self.ax_theta.tick_params(colors='white', labelsize=7)
        for s in self.ax_theta.spines.values(): s.set_color('gray')
        self.line_theta, = self.ax_theta.plot([], [], 'cyan', lw=1.5)
        self.line_dtheta, = self.ax_theta.plot([], [], 'red', lw=0.8, alpha=0.6)
        self.ax_theta.set_ylim(-0.1, np.pi/2 + 0.1)
        self.ax_theta.axhline(np.pi/4, color='white', linestyle=':', alpha=0.3)

        # Row 2: Input, Conf+Mom, Vortex, g, Fingerprint (kept for optional HUD)
        self.ax_in = self.fig.add_subplot(256)
        self.ax_in.set_facecolor('black')
        self.ax_in.set_title('INPUT', color='lime', fontsize=9)
        self.ax_in.tick_params(colors='white', labelsize=7)
        for s in self.ax_in.spines.values(): s.set_color('gray')
        self.line_amp, = self.ax_in.plot([], [], 'g-', lw=0.5, alpha=0.7)
        self.ax_in.set_ylim(0, 0.2)

        self.ax_cm = self.fig.add_subplot(257)
        self.ax_cm.set_facecolor('black')
        self.ax_cm.set_title('CONF + MOM', color='white', fontsize=9)
        self.ax_cm.tick_params(colors='white', labelsize=7)
        for s in self.ax_cm.spines.values(): s.set_color('gray')
        self.line_conf, = self.ax_cm.plot([], [], 'm-', lw=1)
        self.line_mom, = self.ax_cm.plot([], [], 'c-', lw=1, alpha=0.7)
        self.ax_cm.set_ylim(0, 1)

        self.ax_vortex = self.fig.add_subplot(258)
        self.ax_vortex.set_facecolor('black')
        self.ax_vortex.set_title('VORTEX COUNT', color='lime', fontsize=9)
        self.ax_vortex.tick_params(colors='white', labelsize=7)
        for s in self.ax_vortex.spines.values(): s.set_color('gray')
        self.line_vortex, = self.ax_vortex.plot([], [], color='lime', lw=1)
        self.ax_vortex.set_ylim(0, 50)

        self.ax_g = self.fig.add_subplot(259)
        self.ax_g.set_facecolor('black')
        self.ax_g.set_title('NONLINEARITY (g)', color='orange', fontsize=9)
        self.ax_g.tick_params(colors='white', labelsize=7)
        for s in self.ax_g.spines.values(): s.set_color('gray')
        self.line_g, = self.ax_g.plot([], [], color='orange', lw=1)
        self.ax_g.set_ylim(-2, 4)
        self.ax_g.axhline(0, color='white', linestyle='-', alpha=0.5, lw=0.5)

        self.ax_fp = self.fig.add_subplot(2, 5, 10)
        self.ax_fp.set_facecolor('black')
        self.ax_fp.set_title('FINGERPRINT', color='gray', fontsize=9)
        self.ax_fp.set_xticks([]); self.ax_fp.set_yticks([])
        self.im_fp = self.ax_fp.imshow(np.zeros((8, 8)), cmap='plasma', vmin=0, vmax=1)

        # Keypress control
        self.scale_i = 0
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        plt.tight_layout()
        self.fig.canvas.draw()
        plt.show(block=False)

        self._apply_hud_visibility()

    def _apply_hud_visibility(self):
        # Hide the bottom row if not full HUD (keeps it “stage clean”)
        for ax in [self.ax_in, self.ax_cm, self.ax_vortex, self.ax_g, self.ax_fp]:
            ax.set_visible(self.hud_full)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        h = self.h
        k = event.key

        if k == '[':
            h.synth.root_midi -= 1
        elif k == ']':
            h.synth.root_midi += 1
        elif k in ('1', '2', '3', '4'):
            h.synth.max_voices = int(k)
        elif k == 'k':                                  # was "s"
            self.scale_i = (self.scale_i + 1) % len(SCALES)
            name, scale = SCALES[self.scale_i]
            h.synth.set_scale(name, scale)
        elif k == 'h':
            self.hud_full = not self.hud_full
            self._apply_hud_visibility()
        elif k == 'd':
            h.dream_enabled = not h.dream_enabled
            if not h.dream_enabled:
                h.dream_depth = 0.0
                h.is_dreaming = False
        elif k == 'p':
            self.palette_i = (self.palette_i + 1) % len(self.palettes)
            self.im_intensity.set_cmap(self.palettes[self.palette_i])

    def update(self):
        h = self.h
        am = h.attractor_memory
        pm = h.path_memory

        I = np.abs(h.psi)**2
        if self.trail is None:
            self.trail = I.copy()
        else:
            self.trail = self.trail_alpha * self.trail + (1 - self.trail_alpha) * I

        self.im_intensity.set_data(self.trail)
        self.im_intensity.set_clim(0, np.percentile(self.trail, 99) + 0.02)

        rgb = complex_to_rgb(h.psi)
        self.im_phase.set_data(rgb)

        age = (h.total_runtime + (time.time() - h.session_start)) / 3600
        self.txt_age.set_text(f'Age: {age:.2f}h | Interactions: {h.total_interactions + h.session_interactions}')

        # Mode
        if h.surprise > 0.3:
            mode, color = "SURPRISED!", 'red'
        elif h.boredom > 0.7:
            mode, color = "VERY BORED", 'darkgray'
        elif h.boredom > 0.5:
            mode, color = "BORED", 'gray'
        elif h.audio.amplitude > 0.03:
            mode, color = "LISTENING", 'lime'
        elif h.dream_depth > 0.7:
            mode, color = "DEEP DREAM", 'purple'
        elif h.is_dreaming:
            mode, color = "DREAMING", 'violet'
        elif am.is_stable:
            mode, color = "STABLE", 'magenta'
        elif pm.get_momentum() > 0.5:
            mode, color = "TRAVERSING", 'cyan'
        else:
            mode, color = "DRIFTING", 'gray'
        self.txt_mode.set_text(mode)
        self.txt_mode.set_color(color)

        g = h.smoothed_g
        if g >= 0:
            regime_txt = f'g = +{g:.2f} (focus)'
            regime_color = 'orange'
        else:
            regime_txt = f'g = {g:.2f} (defocus)'
            regime_color = 'purple'
        self.txt_regime.set_text(regime_txt)
        self.txt_regime.set_color(regime_color)

        # Mood
        if h.mood > 0.75:
            mood_desc = "Excited"
        elif h.mood > 0.6:
            mood_desc = "Engaged"
        elif h.mood > 0.45:
            mood_desc = "Content"
        elif h.mood > 0.3:
            mood_desc = "Calm"
        else:
            mood_desc = "Drowsy"
        self.txt_mood.set_text(f'Mood: {h.mood:.2f} ({mood_desc})')
        self.txt_mood.set_color('yellow' if h.mood > 0.7 else 'white' if h.mood > 0.5 else 'gray')

        # Key/scale/voices + dream lock
        dream_txt = "ON" if h.dream_enabled else "LOCKED"
        self.txt_music.set_text(
            f"Key root_midi={h.synth.root_midi} | Scale={h.synth.scale_name} | Voices={h.synth.max_voices} | Dream={dream_txt}"
        )

        # Optional misc HUD line (minimal by default)
        if self.hud_full:
            self.txt_misc.set_text(
                f"Surp {h.surprise:.2f} | Bor {h.boredom:.2f} | Vx {h.vortex_count} | f0 {h.synth.primary_freq:.1f}Hz"
            )
        else:
            self.txt_misc.set_text("")

        # Affect traces
        n = len(h.mood_hist)
        if n > 3:
            x = np.arange(n)
            self.line_mood_trace.set_data(x, list(h.mood_hist))
            self.line_surprise_trace.set_data(x, list(h.surprise_hist))
            self.line_boredom_trace.set_data(x, list(h.boredom_hist))
            self.ax_affect.set_xlim(0, max(n, 50))

        # Theta traces
        n_theta = len(h.theta_hist)
        if n_theta > 3:
            x_theta = np.arange(n_theta)
            self.line_theta.set_data(x_theta, list(h.theta_hist))
            dtheta_scaled = [d * 5 + np.pi/4 for d in h.dtheta_hist]
            self.line_dtheta.set_data(x_theta, dtheta_scaled)
            self.ax_theta.set_xlim(0, max(n_theta, 50))

        # Full HUD panels
        if self.hud_full:
            n2 = len(h.audio.amp_hist)
            if n2 > 3:
                x2 = np.arange(n2)
                self.line_amp.set_data(x2, list(h.audio.amp_hist))
                self.ax_in.set_xlim(0, max(n2, 50))

            n3 = len(h.confidence_hist)
            if n3 > 3:
                x3 = np.arange(n3)
                self.line_conf.set_data(x3, list(h.confidence_hist))
                self.line_mom.set_data(x3, list(h.momentum_hist))
                self.ax_cm.set_xlim(0, max(n3, 50))

            n4 = len(h.vortex_hist)
            if n4 > 3:
                x4 = np.arange(n4)
                self.line_vortex.set_data(x4, list(h.vortex_hist))
                self.ax_vortex.set_xlim(0, max(n4, 50))
                max_v = max(h.vortex_hist) if h.vortex_hist else 50
                self.ax_vortex.set_ylim(0, max(50, max_v * 1.1))

            n5 = len(h.g_hist)
            if n5 > 3:
                x5 = np.arange(n5)
                self.line_g.set_data(x5, list(h.g_hist))
                self.ax_g.set_xlim(0, max(n5, 50))

            if am.current_coarse is not None:
                self.im_fp.set_data(am.current_coarse)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print(f"H.E.R.B.I.E. - {J_SERIES}: Jam View")
    print("Harmonic Entrainment Reservoir with Bistable Integrative Emergence")
    print("=" * 70)
    print(f"State directory: {SAVE_DIR}")
    print("Controls (in the plot window):")
    print("  [ / ] : transpose key down/up")
    print("  1..4  : set max voices")
    print("  s     : cycle scale (minor pent / dorian / chromatic / major pent)")
    print("  d     : toggle dream enable (LOCK dream OFF for steady jamming)")
    print("  h     : toggle HUD (minimal/full)")
    print("  p     : cycle intensity palette")
    print("Press Ctrl+C to exit (state will be saved).")
    print("=" * 70)

    h = HERBIE()
    v = Viz(h)
    h.start()

    try:
        while True:
            t0 = time.time()
            h.step()
            v.update()
            time.sleep(max(0, 0.03 - (time.time() - t0)))
    except KeyboardInterrupt:
        print(f"\n[HERBIE {J_SERIES}] Saving memories...")
    finally:
        h.save_state()
        h.stop()
        plt.close('all')

    print(f"[HERBIE {J_SERIES}] Sleeping. Memories preserved.")

if __name__ == '__main__':
    main()
