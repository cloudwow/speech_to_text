import whisper
import sounddevice as sd
import numpy as np
import threading
from pynput import keyboard
from pynput.keyboard import Key, Controller
import queue
import tempfile
import scipy.io.wavfile as wavfile


class SpeechToKeyboard:
    def __init__(self):
        self.keyboard = Controller()
        self.recording = False
        self.audio_queue = queue.Queue()
        self.model = whisper.load_model("base")
        self.sample_rate = 16000

        # Setup keyboard listener for hotkey
        self.key_combination = {keyboard.Key.ctrl_l, keyboard.Key.shift}
        self.current_keys = set()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)

    def on_press(self, key):
        if key in self.key_combination:
            self.current_keys.add(key)
            if all(k in self.current_keys for k in self.key_combination):
                self.start_recording()

    def on_release(self, key):
        if key in self.key_combination:
            self.current_keys.discard(key)
            if not all(k in self.current_keys for k in self.key_combination):
                self.stop_recording()

    def record_audio(self):
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_queue.put(indata.copy())

        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=audio_callback):
            while self.recording:
                sd.sleep(100)

    def transcribe_and_type(self, audio_data):
        # Save audio data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            print(f"Saving audio to {temp_file.name}")
            wavfile.write(temp_file.name, self.sample_rate, audio_data)
            # Transcribe audio using Whisper
            result = self.model.transcribe(temp_file.name)
            text = str(result["text"])  # Ensure text is a string
            print(f"Transcribed text: {text}")
            if text:
                # Type out the transcribed text
                self.keyboard.type(text)

    def start_recording(self):
        if not self.recording:
            print("Recording started... (Hold Ctrl+Shift to continue recording)")
            self.recording = True
            self.audio_data = []

            # Start recording in a separate thread
            self.record_thread = threading.Thread(target=self.record_audio)
            self.record_thread.start()

    def stop_recording(self):
        if self.recording:
            print("Recording stopped. Processing...")
            self.recording = False
            self.record_thread.join()

            # Collect all audio data from queue
            audio_chunks = []
            while not self.audio_queue.empty():
                audio_chunks.append(self.audio_queue.get())

            if audio_chunks:
                audio_data = np.concatenate(audio_chunks)
                self.transcribe_and_type(audio_data)

    def run(self):
        self.listener.start()
        print("Speech to Keyboard started!")
        print("Hold Ctrl+Shift+m to record speech and release to stop")

        self.listener.join()


if __name__ == "__main__":
    app = SpeechToKeyboard()
    app.run()
