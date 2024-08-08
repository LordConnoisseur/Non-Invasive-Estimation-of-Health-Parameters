# Heart Rate Extraction from Video

This project extracts heart rate from video footage of a finger illuminated by a torch light. It processes the video to analyze red intensity changes, identifies peaks corresponding to heartbeats, and calculates the beats per minute (BPM).

## Features

1. Extract red intensity from video frames.
2. Normalize red intensity values.
3. Detect peaks in the intensity signal.
4. Calculate BPM from the detected peaks.
5. Check signal quality to ensure reliable BPM calculation.
6. Plot red intensity with detected peaks.
7. GUI to upload video and display BPM result.

## Requirements

- Python3
- OpenCV
- NumPy
- Matplotlib
- SciPy
- Tkinter

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/LordConnoisseur/Non-Invasive-Estimation-of-Health-Parameters.git
    cd ./Non-Invasive-Estimation-of-Health-Parameters
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the script:
    ```sh
    python3 bpm.py
    ```

2. A GUI window will appear. Click "Select Video" to choose a video file for analysis.

3. The script will process the video and display the BPM result if the signal quality is good.

## Project Members
Sundaresh Karthic Ganesan & Tharunithi TJ