import pyaudio
import wave
import sys

# for testing what sampling i can go down to and what effects it has
class AudioFile:
    chunk = 1024

    def __init__(self, file):
        """ Init audio stream """
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )

    def play(self):
        """ Play entire file """
        data = self.wf.readframes(self.chunk)
        while data != b'':
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)

    def close(self):
        """ Graceful shutdown """
        self.stream.close()
        self.p.terminate()

# Usage example for pyaudio
a = AudioFile("Animal-Sound-Dataset/bird/bird_1.wav")
a.play()
a.close()
a = AudioFile("Animal-Sound-Dataset/cat/cat_1.wav")
a.play()
a.close()