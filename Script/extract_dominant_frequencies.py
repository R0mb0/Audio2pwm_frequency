import sys
import os
import json

# --- Check required modules ---
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
    print("Error: The following Python modules are required but missing:")
    for m in missing:
        print(f"  - {m}")
    print("\nInstall the missing modules with the following command:")
    print(f"pip install {' '.join(missing)}")
    sys.exit(1)

SUPPORTED_EXTENSIONS = ['.wav', '.flac', '.ogg', '.aiff']

def load_settings(settings_path):
    with open(settings_path, 'r') as f:
        return json.load(f)

def dominant_frequency_fft(chunk, samplerate):
    if len(chunk) < 2:
        return 0.0
    chunk = chunk - np.mean(chunk)
    spectrum = np.fft.rfft(chunk)
    freqs = np.fft.rfftfreq(len(chunk), d=1.0/samplerate)
    magnitudes = np.abs(spectrum)
    peak_idx = np.argmax(magnitudes)
    return freqs[peak_idx]

def dominant_frequency_autocorrelation(chunk, samplerate):
    if len(chunk) < 2:
        return 0.0
    chunk = chunk - np.mean(chunk)
    autocorr = np.correlate(chunk, chunk, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    # Find first minimum (to avoid zero lag) and then next peak
    d = np.diff(autocorr)
    start = np.where(d > 0)[0]
    if len(start) == 0:
        return 0.0
    start = start[0]
    peak = np.argmax(autocorr[start:]) + start
    if peak == 0:
        return 0.0
    return samplerate / peak

def dominant_frequency_zcr(chunk, samplerate):
    if len(chunk) < 2:
        return 0.0
    zero_crossings = np.where(np.diff(np.sign(chunk)))[0]
    if len(zero_crossings) < 2:
        return 0.0
    avg_period = np.mean(np.diff(zero_crossings))  # in samples
    if avg_period == 0:
        return 0.0
    freq = samplerate / avg_period
    return freq

def dominant_frequency_cepstrum(chunk, samplerate):
    if len(chunk) < 2:
        return 0.0
    chunk = chunk - np.mean(chunk)
    spectrum = np.fft.fft(chunk)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.ifft(log_spectrum).real
    # ignore the very low quefrency (DC and near-zero)
    min_quefrency = int(samplerate / 1000)  # ignore < 1000 Hz
    peak = np.argmax(cepstrum[min_quefrency:]) + min_quefrency
    if peak == 0:
        return 0.0
    freq = samplerate / peak
    return freq

ALGORITHM_FUNCTIONS = {
    "fft": dominant_frequency_fft,
    "autocorrelation": dominant_frequency_autocorrelation,
    "zcr": dominant_frequency_zcr,
    "cepstrum": dominant_frequency_cepstrum
}

def extract_dominant_frequencies(audio_data, samplerate, samples_per_group, algorithm):
    num_samples = len(audio_data)
    frequencies = []
    func = ALGORITHM_FUNCTIONS.get(algorithm, dominant_frequency_fft)
    for start in range(0, num_samples, samples_per_group):
        end = min(start + samples_per_group, num_samples)
        chunk = audio_data[start:end]
        if len(chunk) == 0:
            continue
        freq = func(chunk, samplerate)
        frequencies.append(freq)
    return frequencies

def get_audio_files():
    files = []
    for file in os.listdir('.'):
        ext = os.path.splitext(file)[1].lower()
        if ext in SUPPORTED_EXTENSIONS and os.path.isfile(file):
            files.append(file)
    return files

def choose_files(files):
    print("Audio files found in the current directory:")
    for idx, fname in enumerate(files):
        print(f"  [{idx}] {fname}")
    print("Choose a file by number, or 'A' to process all files.")
    while True:
        choice = input("Your choice: ").strip()
        if choice.lower() == 'a':
            return files
        if choice.isdigit() and 0 <= int(choice) < len(files):
            return [files[int(choice)]]
        print("Invalid input. Please enter a valid number or 'A'.")

def ensure_output_folder(folder='output'):
    if not os.path.isdir(folder):
        os.makedirs(folder)

def next_available_filename(base, folder, ext=".txt"):
    candidate = os.path.join(folder, base + ext)
    if not os.path.exists(candidate):
        return candidate
    n = 1
    while True:
        candidate = os.path.join(folder, f"{base}{n}{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1

def process_file(audio_path, samples_per_group, algorithm, output_folder):
    try:
        audio_data, samplerate = sf.read(audio_path)
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    frequencies = extract_dominant_frequencies(audio_data, samplerate, samples_per_group, algorithm)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = next_available_filename(base_name, output_folder)
    with open(output_path, 'w') as f:
        f.write(f"# Algorithm used: {algorithm}\n")
        for freq in frequencies:
            f.write(f"{freq:.2f}\n")
    print(f"File '{audio_path}' processed. Output: '{output_path}' (Algorithm: {algorithm})")

def main():
    settings_path = "settings.json"
    output_folder = "output"

    # Check settings.json exists
    if not os.path.isfile(settings_path):
        print(f"Error: {settings_path} not found.")
        sys.exit(1)

    # Load parameters
    settings = load_settings(settings_path)
    samples_per_group = settings.get('samples_per_group', 1024)
    algorithm = settings.get('algorithm', 'fft').lower()

    if samples_per_group < 2:
        print("Error: 'samples_per_group' must be at least 2 to calculate the dominant frequency.")
        sys.exit(1)

    if algorithm not in ALGORITHM_FUNCTIONS:
        print(f"Error: Unknown algorithm '{algorithm}'. Supported algorithms are: {list(ALGORITHM_FUNCTIONS.keys())}")
        sys.exit(1)

    audio_files = get_audio_files()
    if not audio_files:
        print("No supported audio files found in the current directory.")
        sys.exit(1)

    if len(audio_files) == 1:
        print(f"Found only one audio file: {audio_files[0]}. Processing automatically.")
        files_to_process = audio_files
    else:
        files_to_process = choose_files(audio_files)

    ensure_output_folder(output_folder)

    for audio_path in files_to_process:
        process_file(audio_path, samples_per_group, algorithm, output_folder)

if __name__ == "__main__":
    main()