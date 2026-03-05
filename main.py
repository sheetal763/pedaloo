import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier


# Bandpass Filter
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


# Median Frequency
def median_frequency(signal, fs):
    fft_vals = np.abs(rfft(signal))
    freqs = rfftfreq(len(signal), 1/fs)
    cumulative = np.cumsum(fft_vals)
    if cumulative[-1] == 0:
        return 0.0
    indices = np.where(cumulative >= cumulative[-1] / 2)[0]
    if len(indices) == 0:
        return 0.0
    return freqs[indices[0]]


# EMG Processing Function
def analyze_emg(file):

    if file is None:
        return "Please upload a TXT or CSV file", None, None

    try:

        filename = file.name.lower()

        # automatic delimiter detection
        if filename.endswith(".csv"):
            data = pd.read_csv(file.name)

        elif filename.endswith(".txt"):
            data = pd.read_csv(file.name, sep=None, engine="python", header=None)

        else:
            return "Unsupported file format. Upload .txt or .csv", None, None


        # keep numeric columns only
        data = data.select_dtypes(include=[np.number])

        if data.shape[1] == 0:
            return "File must contain at least one numeric column", None, None

        if data.shape[1] == 1:
            # Single column: treat as EMG amplitude, generate time axis
            emg = data.iloc[:,0].values
            fs = 1000  # default sampling frequency (Hz)
            time = np.arange(len(emg)) / fs
        else:
            # Two+ columns: first is time, second is EMG
            time = data.iloc[:,0].values
            emg = data.iloc[:,1].values
            fs = int(1 / np.mean(np.diff(time)))


        # EMG preprocessing
        filtered = bandpass_filter(emg, 20, 450, fs)
        rectified = np.abs(filtered)


        # RMS calculation
        window = int(0.2 * fs)
        rms = np.sqrt(np.convolve(rectified**2,
                                  np.ones(window)/window,
                                  mode='valid'))


        # median frequency
        mf = median_frequency(filtered, fs)


        # fatigue index
        fatigue_index = np.mean(rms) / mf


        # simple ML classifier
        X = np.array([
            [0.10,130,0.0007],
            [0.12,120,0.0010],
            [0.30,70,0.0040],
            [0.28,75,0.0037]
        ])

        y = np.array(["Healthy","Healthy","Impaired","Impaired"])

        model = RandomForestClassifier()
        model.fit(X,y)

        features = np.array([[np.mean(rms), mf, fatigue_index]])

        prediction = model.predict(features)[0]


        # Plot Raw EMG
        fig1, ax1 = plt.subplots()
        ax1.plot(time, emg)
        ax1.set_title("Raw EMG Signal")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")


        # Plot RMS
        fig2, ax2 = plt.subplots()
        ax2.plot(rms)
        ax2.set_title("RMS Envelope")


        result = f"""
File Uploaded Successfully

Sampling Frequency: {fs} Hz

Mean RMS: {np.mean(rms):.4f}

Median Frequency: {mf:.2f} Hz

Fatigue Index: {fatigue_index:.6f}

Muscle Condition Classification: {prediction}
"""


        return result, fig1, fig2


    except Exception as e:
        return f"Processing error: {str(e)}", None, None



# Gradio UI
interface = gr.Interface(
    fn=analyze_emg,
    inputs=gr.File(label="Upload EMG File (.TXT or .CSV)"),
    outputs=[
        gr.Textbox(label="EMG Analysis Result"),
        gr.Plot(label="Raw EMG Signal"),
        gr.Plot(label="RMS Envelope")
    ],
    title="EMG Fatigue Detection and Muscle Health Classification",
    description="Upload a TXT or CSV EMG file containing Time and EMG amplitude columns."
)

interface.launch()