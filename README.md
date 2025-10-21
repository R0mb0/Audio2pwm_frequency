# 🎼 Dominant Frequency Extractor for Arduino PWM

This Python script extracts the dominant frequency sequence from audio files and exports them as text files, ready for Arduino PWM sound reproduction. 
You can choose the extraction algorithm via the `settings.json` file.

## ⚡ Supported Algorithms

- **fft**: Fast Fourier Transform. Good for simple, relatively stationary signals.
- **autocorrelation**: Finds periodicity by correlating the signal with itself. Robust for detecting pitch in periodic signals.
- **zcr**: Zero Crossing Rate. Very simple, works best for pure tones.
- **cepstrum**: Uses the cepstral domain to find pitch, robust for complex signals.

## 🛠️ Configuration

Create a `settings.json` file in the same directory as the script. Example:

```json
{
  "samples_per_group": 300,
  "algorithm": "fft"
}
```

- `samples_per_group`: Number of samples per analysis window (min 2).
- `algorithm`: Choose `"fft"`, `"autocorrelation"`, `"zcr"`, or `"cepstrum"`.

## 🚀 Usage

1. Place your audio files (`.wav`, `.flac`, `.ogg`, `.aiff`) in the script directory.
2. Run the script:  
   `python extract_dominant_frequencies.py`
3. If multiple audio files are present, select one or all.
4. Output `.txt` files will be created in the `output` folder, named after the audio file.

Each output file starts with a comment line indicating the algorithm used.

## 📦 Requirements

- Python 3.x
- `numpy`, `soundfile` (install via `pip install numpy soundfile`)

## 💡 Notes

- For best results, experiment with different algorithms and `samples_per_group` values.
- The output frequencies are suitable for PWM generation in Arduino.
