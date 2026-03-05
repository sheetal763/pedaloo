# EMG Fatigue Detection and Muscle Health Classification

A web-based tool for analyzing Electromyography (EMG) signals to detect muscle fatigue and classify muscle health using signal processing and machine learning.

## Features

- **EMG Signal Upload** — Supports `.txt` and `.csv` file formats with automatic delimiter detection.
- **Signal Preprocessing** — Bandpass filtering (20–450 Hz) and full-wave rectification.
- **RMS Envelope** — Computes the Root Mean Square envelope over a 200 ms sliding window.
- **Median Frequency** — Calculates the median frequency of the filtered signal via FFT.
- **Fatigue Index** — Derives a fatigue index from the ratio of mean RMS to median frequency.
- **ML Classification** — Classifies the muscle condition as **Healthy** or **Impaired** using a Random Forest classifier.
- **Interactive Plots** — Displays the raw EMG signal and RMS envelope through a Gradio web UI.

## Project Structure

| File | Description |
|---|---|
| `main.py` | Gradio web app — EMG analysis pipeline and UI |
| `convert.py` | Converts BioPac `.acq` files to CSV using `bioread` |
| `ABEL-HAM.acq` | Sample BioPac acquisition file |
| `requirements.txt` | Python dependencies |

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - `gradio`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `scikit-learn`
  - `biopython`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Convert a BioPac `.acq` file to CSV (optional)

If your raw data is in BioPac format, convert it first:

```bash
python convert.py
```

This reads `ABEL-HAM.acq` and writes `emg_data.csv` with `time` and `emg` columns.

### 2. Launch the web app

```bash
python main.py
```

A Gradio interface will open in your browser. Upload a `.txt` or `.csv` file containing EMG data.

#### Input format

- **Single column** — Treated as EMG amplitude. A time axis is generated assuming a 1000 Hz sampling rate.
- **Two+ columns** — First column is treated as time (seconds), second as EMG amplitude. Sampling frequency is inferred from the time column.

#### Output

- Sampling frequency, mean RMS, median frequency, fatigue index, and muscle condition classification.
- Raw EMG signal plot.
- RMS envelope plot.

## Flowchart

```mermaid
flowchart TD
    A[Upload EMG File .txt / .csv] --> B[Read & Parse File]
    B --> C{Column Count}
    C -- 1 column --> D[Use as EMG amplitude\nAssume fs = 1000 Hz]
    C -- 2+ columns --> E[Col 1 = Time\nCol 2 = EMG\nInfer fs]
    D --> F[Bandpass Filter 20–450 Hz]
    E --> F
    F --> G[Full-Wave Rectification]
    G --> H[RMS Envelope 200 ms window]
    G --> I[Median Frequency via FFT]
    H --> J[Compute Fatigue Index\nmean RMS / median freq]
    I --> J
    J --> K[Random Forest Classification]
    K --> L{Prediction}
    L -- Healthy --> M[Display Results & Plots]
    L -- Impaired --> M
```

## How It Works

1. **Bandpass Filter** — A 4th-order Butterworth filter isolates the 20–450 Hz EMG frequency band.
2. **Rectification** — The filtered signal is full-wave rectified (absolute value).
3. **RMS Envelope** — A sliding-window RMS is computed over a 200 ms window.
4. **Median Frequency** — The frequency at which the cumulative power spectrum reaches 50%.
5. **Fatigue Index** — `mean(RMS) / median_frequency`. Higher values may indicate fatigue.
6. **Classification** — A Random Forest model trained on reference feature vectors predicts **Healthy** or **Impaired**.
