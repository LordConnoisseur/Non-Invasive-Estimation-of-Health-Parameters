import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
import tkinter as tk
from tkinter import filedialog, messagebox

def extract_red_intensity(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    red_intensities = []
    
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        red_channel = frame[:, :, 2] 
        avg_red_intensity = np.mean(red_channel)
        red_intensities.append(avg_red_intensity)
            
    cap.release()
    return red_intensities, frame_count, fps

def normalize_intensities(intensities):
    min_intensity = np.min(intensities)
    max_intensity = np.max(intensities)
    normalized_intensities = (intensities - min_intensity) / (max_intensity - min_intensity)
    return normalized_intensities

def find_peaks_in_intensity(intensities, distance=30, height=None, threshold=None, prominence=None):
    peaks, properties = find_peaks(intensities, distance=distance, height=height, threshold=threshold, prominence=prominence)
    return peaks, properties

def calculate_bpm(peaks, duration_in_seconds):
    num_beats = len(peaks)
    bpm = (num_beats / duration_in_seconds) * 60
    return bpm

def plot_red_intensity_with_peaks(intensities, peaks, properties, video_name):
    plt.figure(figsize=(12, 6))
    plt.plot(intensities, label='Red Intensity')
    plt.plot(peaks, np.array(intensities)[peaks], "x", label='Peaks')
    plt.title(f'Red Intensity and Peaks over Time for {video_name}')
    plt.xlabel('Frame Number')
    plt.ylabel('Average Red Intensity')
    plt.legend()
    plt.show()

def signal_quality_check(intensities, peaks, properties, fps):
    signal_power = np.mean(np.array(intensities)[peaks] ** 2)
    noise_power = np.var(intensities)
    snr = signal_power / (noise_power + 1e-10) 
    
    valid_peak_count = len(peaks)
    if valid_peak_count < (0.5 * fps): 
        return False, snr
    
    if snr < 2: 
        return False, snr
    
    return True, snr

def process_video_and_calculate_bpm(video_path, video_name):
    intensities, frame_count, fps = extract_red_intensity(video_path)
    normalized_intensities = normalize_intensities(intensities)
    peaks, properties = find_peaks_in_intensity(normalized_intensities, distance=15, height=None, threshold=None, prominence=0.05)
    
    is_valid_signal, snr = signal_quality_check(normalized_intensities, peaks, properties, fps)
    if not is_valid_signal:
        messagebox.showwarning("Invalid Signal", f"The video {video_name} has a poor signal quality (SNR: {snr:.2f}). Please use a better video.")
        return None
    
    duration_in_seconds = frame_count / fps
    
    bpm = calculate_bpm(peaks, duration_in_seconds)
    plot_red_intensity_with_peaks(normalized_intensities, peaks, properties, video_name)
    return bpm

def select_video():
    video_path = filedialog.askopenfilename(
        title="Select a Video File", 
        filetypes=(("MP4 files", "*.mp4"), ("MOV files", "*.MOV"), ("All files", "*.*"))
    )
    if video_path:
        video_name = os.path.basename(video_path)
        bpm = process_video_and_calculate_bpm(video_path, video_name)
        if bpm:
            messagebox.showinfo("BPM Result", f"Video: {video_name}\nBPM: {bpm:.2f}")
    else:
        messagebox.showwarning("No File Selected", "Please select a video file.")

root = tk.Tk()
root.title("Heart Rate from Video")

select_button = tk.Button(root, text="Select Video", command=select_video)
select_button.pack(pady=20)

root.mainloop()