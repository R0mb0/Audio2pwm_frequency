import sys
import os
import json

# --- Controllo moduli necessari ---
missing = []

try:
    import numpy as np
except ImportError:
    missing.append("numpy")

try:
    import soundfile as sf
except ImportError:
    missing.append("soundfile")

if missing:
    print("Errore: mancano i seguenti moduli Python necessari:")
    for m in missing:
        print(f"  - {m}")
    print("\nInstalla i moduli mancanti con il seguente comando:")
    print(f"pip install {' '.join(missing)}")
    sys.exit(1)

def load_settings(settings_path):
    with open(settings_path, 'r') as f:
        return json.load(f)

def extract_dominant_frequencies(audio_data, samplerate, samples_per_group):
    num_samples = len(audio_data)
    frequencies = []

    for start in range(0, num_samples, samples_per_group):
        end = min(start + samples_per_group, num_samples)
        chunk = audio_data[start:end]
        if len(chunk) == 0:
            continue
        # Remove DC offset
        chunk = chunk - np.mean(chunk)
        # FFT
        spectrum = np.fft.rfft(chunk)
        freqs = np.fft.rfftfreq(len(chunk), d=1.0/samplerate)
        magnitudes = np.abs(spectrum)
        # Trova la frequenza dominante (massimo del modulo)
        peak_idx = np.argmax(magnitudes)
        dominant_freq = freqs[peak_idx]
        frequencies.append(dominant_freq)
    return frequencies

def main():
    settings_path = "settings.json"
    audio_path = "input.ogg"
    output_path = "output.txt"

    # Check settings.json exists
    if not os.path.isfile(settings_path):
        print(f"Errore: {settings_path} non trovato.")
        sys.exit(1)

    # Carica parametri
    settings = load_settings(settings_path)
    samples_per_group = settings.get('samples_per_group', 1024)

    # Leggi file audio
    if not os.path.isfile(audio_path):
        print(f"Errore: {audio_path} non trovato.")
        sys.exit(1)
    audio_data, samplerate = sf.read(audio_path)
    # Se stereo, usa solo un canale
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:,0]
    
    # Estrai frequenze dominanti
    frequencies = extract_dominant_frequencies(audio_data, samplerate, samples_per_group)
    
    # Scrivi su file
    with open(output_path, 'w') as f:
        for freq in frequencies:
            f.write(f"{freq:.2f}\n")

    print(f"Frequenze dominanti estratte in {output_path}")

if __name__ == "__main__":
    main()