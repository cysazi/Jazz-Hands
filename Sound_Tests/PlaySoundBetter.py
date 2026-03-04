# imports
import numpy as np
import pyaudio

# defining the class
class SoundEngine:
    # runs automatically when you create engine = SoundEngine()
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.sample_rate = 44100
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=self.sample_rate,
                                  output=True)

        self.frequency = 440.0
        self.amplitude = 0.5
        self.phase = 0.0

    def update_pitch(self, hand_y):
        self.frequency = 200 + (1 - hand_y) * 800

    def generate_step(self, frames = 512):
        t = (np.arange(frames) + self.phase) / self.sample_rate
        wave = self.amplitude * np.sin(2 * np.pi * self.frequency * t)

        self.phase += frames
        return wave.astype(np.float32).tobytes()

    def play(self):
        data = self.generate_step()
        self.stream.write(data)
