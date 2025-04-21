#importing necessary libraries
import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
import os

# Path to the saved model
MODEL_PATH = "scream_detector_model.pkl"

# Constants
SAMPLE_RATE = 16000
# time in seconds for recording and well as input 
DURATION = 10
N_MFCC = 13

# Loading the pre-trained classifier model
clf = joblib.load(MODEL_PATH)
classes = {0: "Ambient", 1: "Scream", 2: "Conversation"}

# Feature extraction function
def extract_features(file_path):
    """Extract MFCC features from the given audio file."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
    if len(y) < SAMPLE_RATE * DURATION:
        y = np.pad(y, (0, int(SAMPLE_RATE * DURATION - len(y))))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc, axis=1).reshape(1, -1)

# prediction function
def predict_audio(file_path):
    """Predict the class (Ambient, Scream, Conversation) of the audio file."""
    try:
        features = extract_features(file_path)
        prediction = clf.predict(features)[0]
        return f"Prediction: {classes[prediction]}"
    except Exception as e:
        return f"Error: {e}"

# GUI 
def upload_file():
    # For handling file upload and prediction
    filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if filepath:
        result = predict_audio(filepath)
        result_label.config(text=result)

# for recording live audio
def record_audio():
    output_file = "recorded_input.wav"
    try:
        result_label.config(text=" Recording... ")
        app.update()  # Update the GUI to show the recording text
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        # Wait function until recording is finished
        sd.wait()
         # Save the recording to a file
        write(output_file, SAMPLE_RATE, recording) 

        result = predict_audio(output_file)
        # Clean up by removing that temporary file
        os.remove(output_file)  
        result_label.config(text=result)
    except Exception as e:
        result_label.config(text=f"Recording failed !!!!: {e}")

# User Interface 
app = tk.Tk()
app.title("Scream Detector")
app.geometry("400x250")
app.configure(bg="#f0f0f0")

title = tk.Label(app, text="Audio Classifier", font=("Arial", 18), bg="#f0f0f0")
title.pack(pady=10)

upload_btn = tk.Button(app, text=" Upload a WAV File", command=upload_file, font=("Arial", 12),
                       bg="#4caf50", fg="white", activebackground="#45a049")
upload_btn.pack(pady=10)

record_btn = tk.Button(app, text=" Record Live Audio (10s)", command=record_audio, font=("Arial", 12),
                       bg="#2196f3", fg="white", activebackground="#1976d2")
record_btn.pack(pady=10)

result_label = tk.Label(app, text="", font=("Arial", 14), bg="#f0f0f0")
result_label.pack(pady=20)

app.mainloop()
