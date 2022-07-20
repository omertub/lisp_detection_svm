import matplotlib.pyplot as plt
from matplotlib import cm
import wave
import sys
from scipy.io.wavfile import read
from python_speech_features import mfcc
from sklearn.preprocessing import normalize
import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
import os

def remove_noise(signal):
    threshold_start = 0.07 * max(signal)
    threshold_end = 0.12 * max(signal)

    for i, s in enumerate(signal):
        if abs(s) > threshold_start:
            break
    start_speech_idx = i
    # signal = np.array([s if abs(s) > noise_threshold or index > i else 0 for index, s in enumerate(signal)])

    for i, s in enumerate(np.flip(signal)):
        if abs(s) > threshold_end:
            break
    end_speech_idx = len(signal) - i
    # signal = np.array([s if index < end_speech_idx else 0 for index, s in enumerate(signal)])
    # signal = np.array([s if abs(s) > noise_threshold or index < i else 0 for index, s in enumerate(signal)])
 
    start_speech_idx = max(0,start_speech_idx - 1000)
    end_speech_idx = min(len(signal), end_speech_idx + 1000)
    # signal = signal[start_speech_idx:start_speech_idx+40000]
    # signal = signal[start_speech_idx:i]
    # signal = np.concatenate((signal[start_speech_idx:], np.zeros(start_speech_idx)))  # pad with zeros to 2 seconds
    return signal[start_speech_idx:end_speech_idx]


def normalize_samples(samples):
    samples = samples.reshape(-1, 1)
    samples = normalize(samples, norm='max', axis=0)
    samples = samples.reshape(-1)
    return samples


def generate_features(wav_file):
    (frame_rate, samples) = read(wav_file)
    samples_n = normalize_samples(samples)
    mfcc_feat = mfcc(signal=samples_n, samplerate=frame_rate,
                     winlen=0.020, winstep=0.010, numcep=24, nfilt=40, nfft=1024, lowfreq=113, highfreq=6854)
    return mfcc_feat

# ספה
def find_cut0(signal_r):
    envelope = gaussian_filter1d(np.abs(hilbert(signal_r)), 500)
    gradient = np.gradient(envelope)
    max_val=np.max(envelope)
    eps=0.0001
    idx = len(envelope) - 1
    for i in range(len(envelope)-100, -1, -1):
        if (gradient[i] > (-eps)) and (envelope[i] > 0.50 * max_val):
            break
    idx = i    
    for i in range(idx-100, -1, -1):
        if (gradient[i] < eps) and (envelope[i] < 0.30 * max_val):
            break

    return i-1000

# גזר
def find_cut5(signal_r):
    envelope = gaussian_filter1d(np.abs(hilbert(signal_r)), 500)
    idx_max = np.argmax(envelope)
    gradient = np.gradient(envelope)
    eps=0.01
    for i in range(idx_max + 100, len(envelope)):
        if gradient[i] > (-eps):
            break

    return i-1500

def plot_wav(file, counter):
    spf = wave.open(file, "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, np.int16)
    # signal_res = remove_noise(signal)
    # signal = normalize_samples(signal) #FIXME: enable once done
    signal_r = remove_noise(signal)
    # signal_r = signal
    # signal_fft = np.fft.fft(signal)
    # signal_fft = np.concatenate((signal_fft[int(len(signal_fft)/2):], signal_fft[:int(len(signal_fft)/2)]))
    # signal_r_fft = np.fft.fft(signal_r)
    # signal_r_fft = np.concatenate((signal_r_fft[int(len(signal_r_fft) / 2):], signal_r_fft[:int(len(signal_r_fft) / 2)]))

    # If Stereo
    if spf.getnchannels() == 2:
        print("Just mono files")
        sys.exit(0)

    plt.figure(1)
    plt.subplot(221)
    plt.title("Signal Wave")
    plt.plot(signal)

    plt.subplot(223)
    plt.title("Signal STFT")
    plt.specgram(signal, Fs=44100)
    # plt.plot(signal_fft)

    plt.subplot(222)
    plt.title("Signal remove noise")
    plt.plot(signal_r, '-')
    envelope = gaussian_filter1d(np.abs(hilbert(signal_r)), 500)
    plt.plot(envelope, 'k')
    
    cut = find_cut0(signal_r)
    plt.axvline(x=cut, color='r')

    plt.subplot(224)
    plt.title("Signal with noise removal FFT")
    plt.specgram(signal_r)
    # plt.plot(signal_r_fft)
    
    plt.show()

    from scipy.io.wavfile import write
    samplerate = 44100
    print(f"tst/e_{counter}.wav")
    if BEFORE:
        write(f"tst/e_{counter}.wav", samplerate, signal_r[:cut].astype(np.int16))
    else:
        write(f"tst/e_{counter}.wav", samplerate, signal_r[cut:].astype(np.int16))


def plot_mfcc(file):
    mfcc_data = generate_features(file)

    fig, ax = plt.subplots()
    mfcc_data= np.swapaxes(mfcc_data, 0, 1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    ax.set_title('MFCC')

    plt.show()

def get_files_for_word(word):
    home_dir = "C:/Users/omer_/Desktop/recording"

    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(home_dir) for f in filenames if os.path.splitext(f)[1] == '.wav']
    files = [item.replace('\\','/') for item in result if word in item]
    return files

WORDS = (
    "ספה",
    "פנס", 
    "קופסה", 
    "זחל",
    "רמז",
    "גזר",
)
BEFORE=1

def main():
    counter=0
    for file in get_files_for_word(WORDS[0]):
        print(file)
        plot_wav(file, counter)
        counter += 1
    # file = "./גזר/bad/5_b0.wav"
    # file = f'./{WORDS[0]}/bad/0_b0.wav'
    # plot_wav(file, counter)
    # plot_mfcc(file)

if __name__ == '__main__':
    main()


# from scipy.io.wavfile import write
# import wave
# import numpy as np
# rec_path = "C:/Users/omer_/Desktop/recording/Itav/ספה/good/0_g1.wav"
# spf = wave.open(rec_path, "r")
# signal = spf.readframes(-1)
# signal = np.frombuffer(signal, np.int16)
# signal = signal[:int(len(signal) / 2)]
# samplerate = 44100
# write(rec_path, samplerate, signal.astype(np.int16))
