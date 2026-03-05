import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --------------------- helper functions ---------------------

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Return a bandpass filtered version of *signal*.

    Uses a Butterworth filter with the given low/high cut-offs in Hz and
    sampling frequency *fs*.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def median_frequency(signal, fs):
    """Compute the median frequency of the power spectrum of *signal*."""
    fft_vals = np.abs(rfft(signal))
    freqs = rfftfreq(len(signal), 1 / fs)
    cumulative = np.cumsum(fft_vals)
    idx = np.where(cumulative >= cumulative[-1] / 2)[0][0]
    return freqs[idx]


def compute_rms(rectified, fs, window_ms=200):
    """Compute the RMS envelope using a sliding window in milliseconds."""
    window = int((window_ms / 1000) * fs)
    if window < 1:
        window = 1
    return np.sqrt(np.convolve(rectified ** 2, np.ones(window) / window, mode="valid"))


def extract_features(rms, filtered, fs):
    """Return mean RMS, RMS std and median frequency as a feature vector."""
    feat_mean = np.mean(rms)
    feat_std = np.std(rms)
    feat_mf = median_frequency(filtered, fs)
    return np.array([feat_mean, feat_std, feat_mf])


def train_synthetic_model():
    """Train and return a simple RandomForest on toy data.

    In real applications replace this with loading a pretrained model from
    disk or retraining on a larger dataset.
    """
    X = np.array([
        [0.10, 0.02, 130],
        [0.12, 0.03, 120],
        [0.30, 0.08, 70],
        [0.28, 0.07, 75],
    ])
    y = np.array(["Healthy", "Healthy", "Impaired", "Impaired"])
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model


# --------------------- application logic ---------------------

def main():
    st.title("EMG Signal Classifier")
    st.write(
        "Upload a CSV file containing two columns: `time` and `emg`. The app will"
        " preprocess the data, extract features, and classify the signal as either"
        " **Healthy** or **Impaired**."
    )

    uploaded = st.file_uploader("Choose EMG CSV", type="csv")
    if uploaded is not None:
        # read into dataframe
        data = pd.read_csv(uploaded)
        required = {"time", "emg"}
        if not required.issubset(data.columns):
            st.error("CSV must contain 'time' and 'emg' columns.")
            return

        time = data["time"].values
        emg = data["emg"].values
        fs = 1000  # assume 1 kHz sampling, adjust as needed

        # preprocessing
        filtered = bandpass_filter(emg, 20, 450, fs)
        rectified = np.abs(filtered)
        rms = compute_rms(rectified, fs)

        # feature extraction & prediction
        features = extract_features(rms, filtered, fs)
        classifier = train_synthetic_model()
        prediction = classifier.predict(features.reshape(1, -1))[0]

        # display results
        st.subheader("Classification Result")
        st.write(f"**Predicted condition:** {prediction}")

        # plotting
        st.subheader("Signal plots")
        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs[0].plot(time, emg, color="tab:blue")
        axs[0].set(title="Raw EMG", ylabel="Amplitude")
        axs[1].plot(time, filtered, color="tab:orange")
        axs[1].set(title="Filtered EMG (20–450 Hz)", ylabel="Amplitude")
        # RMS timeline
        if len(rms) > 0:
            rms_time = np.linspace(time[0], time[-1], len(rms))
            axs[2].plot(rms_time, rms, color="tab:red")
        axs[2].set(title="RMS Envelope", xlabel="Time (s)", ylabel="RMS")
        plt.tight_layout()
        st.pyplot(fig)

        # optional spectrum
        if len(filtered) > 0:
            spec_fig, spec_ax = plt.subplots(figsize=(6, 3))
            fft_vals = np.abs(rfft(filtered))
            freqs = rfftfreq(len(filtered), 1 / fs)
            spec_ax.plot(freqs, fft_vals, color="tab:purple")
            mf_val = median_frequency(filtered, fs)
            spec_ax.axvline(mf_val, color="k", linestyle="--",
                            label=f"Median freq = {mf_val:.1f} Hz")
            spec_ax.set(xlabel="Frequency (Hz)", ylabel="Magnitude",
                        title="Power Spectrum")
            spec_ax.legend()
            spec_ax.set_xlim(0, 500)
            st.pyplot(spec_fig)


if __name__ == "__main__":
    main()
