import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, istft
import pandas as pd


def q1(audio_path) -> np.array:
    """
    param: audio_path: path to q1 audio file
    return: return q1 denoised version
    """
    sample_rate, audio_data = wavfile.read(audio_path)
    fft_audio_data = np.fft.fft(audio_data)

    # create_fft_plot(magnitude, sample_rate)
    # sort_freqs_by_magnitude(magnitude, sample_rate)
    # create_spectrogram(audio_data, sample_rate, 'q1/spectrogram.png')

    argmax_magnitude = np.argmax(np.abs(fft_audio_data))

    fft_audio_data[argmax_magnitude] = 0
    fft_audio_data[-argmax_magnitude] = 0

    # real component
    ifft_audio_data = np.fft.ifft(fft_audio_data).real
    # output_wav("q1/output.wav", sample_rate, ifft_audio_data)

    return ifft_audio_data


def q2(audio_path) -> np.array:
    """
    :param audio_path: path to q2 audio file
    :return: return q2 denoised version
    """
    sample_rate, audio_data = wavfile.read(audio_path)
    frequencies, time, stft_audio = stft(audio_data, sample_rate, nperseg=200)

    # create_spectrogram(audio_data, sample_rate, "q2/spectrogram.jpg")

    rows_names = [str(x) for x in np.int64(frequencies)]
    col_names = [str(x) for x in np.round(time, decimals=3)]
    stft_audio = pd.DataFrame(stft_audio, index=rows_names, columns=col_names)

    # zero frequency 580 - 660 at the timestamp of 1.6 - 4.02 seconds
    stft_audio.loc["580":"640", "1.6":"4.00"] = 0
    time, istft_audio = istft(stft_audio)

    # output_wav("q2/output.wav", sample_rate, istft_audio)

    return istft_audio


def sort_freqs_by_magnitude(magnitude, sample_rate):
    """
    Function to check the highest frequencies present in the magnitude spectrum of a signal.
    DataFrame containing frequency-magnitude tuples sorted in descending order of magnitude.
    :param magnitude: numpy array representing the magnitude spectrum of the signal.
    :param sample_rate: Sampling rate of the signal.
    """
    frequency_axis = np.fft.fftfreq(magnitude.shape[0], 1 / sample_rate)

    # minus value is for sort in descending order
    sorted_indexes = np.argsort(-magnitude)
    sorted_values = magnitude[sorted_indexes]

    frequency_magnitude_tuples = []

    for index, magnitude_value in zip(sorted_indexes, sorted_values):
        frequency_magnitude_tuples.append((frequency_axis[index], np.round(magnitude_value, decimals=2)))

    sorted_magnitude_freq = pd.DataFrame(np.array(frequency_magnitude_tuples, dtype=float), index=['Frequency', 'Magnitude'])


def create_fft_plot(magnitude, sample_rate):
    frequency_axis = np.fft.fftfreq(magnitude.shape[0], 1 / sample_rate)
    plt.plot(frequency_axis, magnitude)
    plt.title('Magnitude per Frequency - DFT Result')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.savefig('q1/dft.png', dpi=200)


def create_spectrogram(audio_data, sample_rate, file_name):
    frequencies, times, spectrogram_data = stft(audio_data, sample_rate, nperseg=200)
    plt.pcolormesh(times, frequencies, (np.abs(spectrogram_data)))
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (Seconds)')
    plt.title('Spectrogram')
    plt.colorbar(label='Magnitude')
    plt.savefig(file_name, dpi=200)


def output_wav(file_name, sample_rate, denoised_audio_data):
    wavfile.write(file_name, sample_rate, denoised_audio_data)

