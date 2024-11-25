import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks, butter, filtfilt, detrend
import os
import tkinter as tk
from tkinter import filedialog, messagebox

def extract_red_intensity(video_path):
    cap = cv2.VideoCapture(video_path)
    red_intensities = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        red_channel = frame[:, :, 2] 
        avg_red_intensity = np.mean(red_channel)
        red_intensities.append(avg_red_intensity)
    cap.release()
    return red_intensities

def preprocess_signal(intensities, fps):
    detrended_signal = detrend(intensities)
    nyquist = 0.5 * fps
    low = 0.5 / nyquist
    high = 4 / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, detrended_signal)
    return filtered_signal

def calculate_bpm(peaks, frame_count, fps):
    num_beats = len(peaks)
    duration_in_seconds = frame_count / fps
    bpm = (num_beats / duration_in_seconds) * 60
    return bpm

def extract_bpm(video_path):
    intensities = extract_red_intensity(video_path)
    fps = 30 
    filtered_signal = preprocess_signal(intensities, fps)
    peaks, _ = find_peaks(filtered_signal, distance=fps//2, prominence=0.01)
    return calculate_bpm(peaks, len(intensities), fps)

def load_video_frames(video_path, num_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (64, 64))
        frames.append(frame_resized[:, :, 2] / 255.0)  # Red channel normalization
        count += 1
    cap.release()
    frames = np.stack(frames, axis=0)
    return torch.tensor(frames, dtype=torch.float32).unsqueeze(0)

class BPRegressionModel(nn.Module):
    def __init__(self):
        super(BPRegressionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(input_size=32 * 16 * 16, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 2) 
        )

    def forward(self, x, bpm):
        batch_size, channels, frames, height, width = x.size()
        cnn_out = self.cnn(x).view(batch_size, frames, -1)
        lstm_out, _ = self.lstm(cnn_out)
        lstm_last_out = lstm_out[:, -1, :] 
        combined = torch.cat((lstm_last_out, bpm.view(-1, 1)), dim=1)
        output = self.fc(combined)
        return output


def load_model(model_path='bp_model_lstm.pth'):
    model = BPRegressionModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_bp(model, video_path):
    bpm = extract_bpm(video_path)
    video_data = load_video_frames(video_path)
    with torch.no_grad():
        prediction = model(video_data.unsqueeze(0), torch.tensor([bpm]).float())
    systolic, diastolic = prediction.squeeze().tolist()
    return systolic, diastolic


if __name__ == "__main__":
    model = load_model('bp_model_lstm.pth')
    
    root = tk.Tk()
    root.withdraw()
    video_file = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
    
    if video_file:
        systolic, diastolic = predict_bp(model, video_file)
        print(f"Predicted Systolic: {systolic:.2f}, Diastolic: {diastolic:.2f}")
    else:
        messagebox.showwarning("No File Selected", "Please select a video file.")