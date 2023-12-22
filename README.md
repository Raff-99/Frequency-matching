import numpy as np
import pyaudio
import matplotlib.pyplot as plt

def get_audio_input(device_index, chunk_size=1024, sample_rate=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=chunk_size)
    return p, stream

def plot_waveform(data):
    plt.plot(data)
    plt.title("Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()

def find_peak_frequency(data, sample_rate):
    # Compute the FFT (Fast Fourier Transform) to find the frequency components
    fft_result = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(fft_result), 1 / sample_rate)

    # Find the index of the peak frequency
    peak_index = np.argmax(np.abs(fft_result))
    peak_frequency = np.abs(frequencies[peak_index])

    return peak_frequency

def tune_guitar_realtime():
    device_index = 1  # Adjust this value to match your microphone's device index

    p, stream = get_audio_input(device_index)

    try:
        print("Tuning your guitar... Press Ctrl+C to stop.")

        while True:
            # Read audio input
            data = np.frombuffer(stream.read(chunk_size), dtype=np.int16)

            # Find the peak frequency in the audio input
            peak_frequency = find_peak_frequency(data, sample_rate)

            # Print the detected frequency in real-time
            print(f"Detected Frequency: {peak_frequency:.2f} Hz", end='\r')

    except KeyboardInterrupt:
        print("\nTuning stopped.")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    chunk_size = 1024
    sample_rate = 44100
    tune_guitar_realtime()

'''/#import threading

def start_realtime_plot():
    tune_thread = threading.Thread(target=tune_guitar_realtime_with_plot)
    tune_thread.start()

if __name__ == "__main__":
    chunk_size = 1024
    sample_rate = 44100

    start_realtime_plot()

    # You can add any additional code here if needed
    input("Press Enter to exit...")'''
