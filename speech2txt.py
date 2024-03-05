from numpy import frombuffer, int16
from pyaudio import PyAudio, paInt16
from faster_whisper import WhisperModel
import audioop

silence_thresh = 300
max_duration = 30
max_silence_seconds = 1.5
model_size = "medium.en"

debug_mode = False 
listen_mode = False

class STT:
    def __init__(
        self,
        max_silence_seconds=max_silence_seconds,
        silence_threshold=silence_thresh,
        chunk=1024,
        sample_format=paInt16,
        channels=1,
        fs=16000,
        max_seconds=max_duration,
    ):
        if debug_mode:
            print("Initializing Ears")
        self.model = WhisperModel(
            model_size, device="cuda", compute_type="float16"
        )
        self.chunk = chunk
        self.sample_format = sample_format
        self.channels = channels
        self.fs = fs
        self.max_seconds = max_seconds
        self.silence_threshold = silence_threshold
        self.max_silence_seconds = max_silence_seconds
        self.p = PyAudio()

    def listen(self):
        frames = []
        self.stream = self.p.open(
            format=self.sample_format,
            channels=self.channels,
            rate=self.fs,
            frames_per_buffer=self.chunk,
            input=True,
        )
        print("Listening...")

        ct = 0
        above_threshold_detected = False
        while True:
            data = self.stream.read(self.chunk)
            frames.append(data)
            rms = audioop.rms(data, 2)
            if not above_threshold_detected and rms < silence_thresh:
                frames.pop()

            if rms > silence_thresh:
                above_threshold_detected = True

            if above_threshold_detected:
                if rms < self.silence_threshold:
                    ct += 1
                else:
                    ct = 0

            if debug_mode:
                print(f"Threshold: {rms}")
            if listen_mode:
                print(f"Hearing: {ct}")

            if ct == 16 * self.max_silence_seconds:
                break

        above_threshold_detected = False

        self.stream.stop_stream()
        self.stream.close()
        print("Recording completed.")
        return frames

    def transcribe(self, frames):
        audio_data = frombuffer(b"".join(frames), dtype=int16)
        audio_data = audio_data.astype("float32") / 32767.0
        segments, info = self.model.transcribe(audio_data, beam_size=5, vad_filter=True)
        return segments

class TTS:
    def speak(self, text, rate=170, volume=1):
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

if __name__ == "__main__":
    obj = STT()
    try:
        while True:
            voice = obj.listen()
            segments = obj.transcribe(voice)
            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    except KeyboardInterrupt:
        pass
