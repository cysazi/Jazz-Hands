from __future__ import annotations
import numpy as np

try:
    import pyaudio
except ImportError:
    pyaudio = None

class SoundEngine:
    def __init__(self, sample_rate: int = 44100):
        if pyaudio is None:
            raise ImportError(
                "pyaudio is required for SoundEngine. Install pyaudio."
            )

        self.p = pyaudio.PyAudio()
        self.sample_rate = sample_rate

        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=self.sample_rate,
            output=True,
        )

    def _midi_to_freq(self, midi_note: int) -> float:
        """MIDI note to frequency"""
        return 440.0 * (2.0 ** ((midi_note - 69)/ 12.0))

    def _get_waveform(self, instrument: str, t: np.ndarray, freq:float) -> np.ndarray:
        """Generates different timbres (instruments)"""
        phase = 2 * np.pi * freq * t

        if instrument == "Synthesizer":
            return (np.sin(phase) + 0.5 * np.sign(np.sin(phase))) / 1.5 # Hybrid
        elif instrument in ["Guitar", "Violin"]:
            return 2 * (phase / (2 * np.pi) % 1) - 1 # Sawtooth (rich harmonics)
        elif instrument == "Flute":
            return np.sin(phase) # Pure Sine
        else: # Piano aka Default
            return np.sign(np.sin(phase)) * 0.5

    def generate_note(
            self,
            note: int,
            volume: int,
            stereo: float,
            attack: int,
            instrument: str,
            duration_ms: int = 500
    ) -> np.ndarray:
        """Generates note and stereo"""
        num_samples = int(self.sample_rate * (duration_ms / 1000))
        t = np.linspace(0, duration_ms / 1000, num_samples, False)

        freq = self._midi_to_freq(note)
        wave = self._get_waveform(instrument, t, freq)

        amp = (volume / 127.0)

        attack_samples = int((attack / 127.0) * num_samples * 0.5)
        envelope = np.ones(num_samples)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-100:] *= np.linspace(1, 0, 100)

        wave = wave *envelope * amp

        # stereo: -1.0 (left to 1.0 (right)
        left_gain = self.clamp(1.0 - stereo, 0.0, 1.0)
        right_gain = self.clamp(1.0 + stereo, 0.0, 1.0)

        stereo_wave = np.zeros((num_samples, 2), dtype=np.float32)
        stereo_wave[:, 0] = wave * left_gain # left channel
        stereo_wave[:, 1] = wave * right_gain # right channel

        return stereo_wave

    def play_data_row(self, row_str: str):
        try:
            note, vol, stereo, attack, inst, reverb = row_str.split(',')

            data = self.generate_note(
                note=int(note),
                volume=int(vol),
                stereo=float(stereo),
                attack=int(attack),
                instrument=inst
            )

            self.stream.write(data)
        except ValueError:
            pass # skip header or malformed rows

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    @staticmethod
    def clamp(value: float, min_v: float, max_v: float) -> float:
        return max(min_v, min(max_v, value))

import wave

def save_to_wav(filename: str, audio: np.ndarray, sample_rate: int = 44100):
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(filename, 'w') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio_int16.tobytes())

def render_song(raw_data: str, engine: SoundEngine) -> np.ndarray:
    chunks = []

    for line in raw_data.strip().split('\n'):
        try:
            note, vol, stereo, attack, inst, reverb = line.split(',')

            chunk = engine.generate_note(
                note=int(note),
                volume=int(vol),
                stereo=float(stereo),
                attack=int(attack),
                instrument=inst
            )

            chunks.append(chunk)

        except ValueError:
            continue

    return np.vstack(chunks)