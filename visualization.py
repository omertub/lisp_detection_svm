import wave, sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from python_speech_features import mfcc
from sklearn.preprocessing import normalize
from scipy.io.wavfile import read, write
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

def remove_noise(signal):
    threshold_start = 0.07 * max(signal)
    threshold_end = 0.12 * max(signal)

    for i, s in enumerate(signal):
        if abs(s) > threshold_start:
            break
    start_speech_idx = i

    for i, s in enumerate(np.flip(signal)):
        if abs(s) > threshold_end:
            break
    end_speech_idx = len(signal) - i
 
    start_speech_idx = max(0,start_speech_idx - 1000)
    end_speech_idx = min(len(signal), end_speech_idx + 1000)

    return signal[start_speech_idx:end_speech_idx]


def normalize_samples(samples):
    samples = samples.reshape(-1, 1)
    samples = normalize(samples, norm='max', axis=0)
    samples = samples.reshape(-1)
    return samples


def generate_features(word_data, wav_file):
    (frame_rate, samples) = read(wav_file)
    samples_n = normalize_samples(samples)
    signal_r = remove_noise(samples_n)

    cut = word_data.cut_func(signal_r)
    signal_r = signal_r[:cut] if (word_data.before) else signal_r[cut:]

    mfcc_feat = mfcc(signal=signal_r, samplerate=frame_rate,
                     winlen=0.020, winstep=0.010, numcep=24, nfilt=40, nfft=1024, lowfreq=113, highfreq=6854)
    return mfcc_feat

# Cut functions:
def find_cut1():
    pass
def find_cut2():
    pass
def find_cut4():
    pass


# ספה
def find_cut0(signal_r):
    envelope = gaussian_filter1d(np.abs(hilbert(signal_r)), 500)
    gradient = np.gradient(envelope)
    max_val = np.max(envelope)
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

# זחל
def find_cut3(signal_r):
    envelope = gaussian_filter1d(np.abs(hilbert(signal_r)), 500)
    gradient = np.gradient(envelope)
    max_val = np.max(envelope)
    eps = 0.0001
    # find max
    for i in range(len(envelope)):
        if (gradient[i] < eps) and (envelope[i] > 0.75 * max_val):
            break
    idx = i
    # find end
    for i in range(idx + 100, len(envelope)):
        if (gradient[i] > -eps) and (envelope[i] < 0.3 * max_val):
            break
    for i in range(idx + 100, len(envelope)):
        if (gradient[i] > -eps) and (envelope[i] < 0.3 * max_val):
            break

    return i - 500

# גזר
def find_cut5(signal_r):
    envelope = gaussian_filter1d(np.abs(hilbert(signal_r)), 500)
    gradient = np.gradient(envelope)
    max_val = np.max(envelope)
    eps=0.0001
    idx = len(envelope) - 1
    for i in range(len(envelope)-100, -1, -1):
        if (gradient[i] > (-eps)) and (envelope[i] > 0.35 * max_val):
            break
    idx = i    
    for i in range(idx-100, -1, -1):
        if (gradient[i] < eps) and (envelope[i] < 0.35 * max_val):
            break

    return i

def plot_wav(file, counter, word_data, normalize):
    spf = wave.open(file, "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, np.int16)

    if normalize:
        signal = normalize_samples(signal)
    signal_r = remove_noise(signal)
 
    # If Stereo
    if spf.getnchannels() == 2:
        print("Just mono files")
        sys.exit(0)

    plt.figure(1)
    plt.subplot(221)
    plt.title("Signal Wave", fontsize=30)
    plt.plot(signal)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(223)
    plt.title("Signal STFT")
    plt.specgram(signal, Fs=44100)

    plt.subplot(222)
    plt.title("Identify Sibilant Syllable", fontsize=30)
    plt.plot(signal_r, '-')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    envelope = gaussian_filter1d(np.abs(hilbert(signal_r)), 500)
    plt.plot(envelope, 'k')
    
    cut = word_data.cut_func(signal_r)
    plt.axvline(x=cut, color='r')
    plt.axvline(x=len(signal_r) - 1, color='r')

    plt.subplot(224)
    plt.title("Signal with noise removal FFT")
    plt.specgram(signal_r)
    
    plt.show()

    print(f"tst/e_{counter}.wav")
    if word_data.before:
        write(f"tst/e_{counter}.wav", SAMPLERATE, signal_r[:cut].astype(np.int16))
    else:
        write(f"tst/e_{counter}.wav", SAMPLERATE, signal_r[cut:].astype(np.int16))


def plot_mfcc(word_data, file):
    mfcc_data = generate_features(word_data, file)

    fig, ax = plt.subplots()
    mfcc_data= np.swapaxes(mfcc_data, 0, 1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    ax.set_title('MFCC')

    plt.show()

def get_files_for_word(dir, word):
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dir) for f in filenames if os.path.splitext(f)[1] == '.wav']
    files = [item.replace('\\','/') for item in result if word in item]
    return files

class word_data:
    def __init__(self, word, before, cut_func):
        self.word = word
        self.before = before
        self.cut_func = cut_func



# creating WORDS       
WORDS = [] 
  
# appending instances to WORDS 
WORDS.append( word_data("ספה",   True,   find_cut0) )
WORDS.append( word_data("פנס",   False,  find_cut5) )
WORDS.append( word_data("קופסה", False,  find_cut5) )
WORDS.append( word_data("זחל",   True,   find_cut3) )
WORDS.append( word_data("רמז",   False,  find_cut4) )
WORDS.append( word_data("גזר",   False,  find_cut5) )

SAMPLERATE = 44100


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        print("Not a dir: ", string)
        exit(-1)

def wav_path(string):
    if string[-3:] != '.wav':
        return string
    else:
        print("Not a wav file: ", string)
        exit(-1)

def main():
    # argparse
    parser = argparse.ArgumentParser(description='Visualize wav files')
    parser.add_argument('-w', '--word', help='Word num to visualize', type=int, required=True)
    parser.add_argument('-m', '--mfcc', help='Visualize mfcc')
    parser.add_argument('-n', '--normalize', help='Normalize signals')
    parser.add_argument('-s', '--signal', help='Visualize signal')
    parser.add_argument('-f', '--file', help='Path to wav file', type=wav_path)
    parser.add_argument('-d', '--dir', help='Path to directory containing wav files', type=dir_path)

    args = parser.parse_args()
    word_data = WORDS[args.word]

    counter=0
    if args.dir:
        for file in get_files_for_word(args.dir, word_data.word):
            print(file)
            if args.signal:
                plot_wav(file, counter, word_data, args.normalize)
            if args.mfcc:
                plot_mfcc(word_data, file)
            counter += 1

    elif args.file:
        file = args.file
        print(file)
        if args.signal:
            plot_wav(file, counter, word_data, args.normalize)
        if args.mfcc:
            plot_mfcc(word_data, file)
    else:
        print("Please use dir (-d=*) or file (-f=*)")
        exit(-1)

if __name__ == '__main__':
    main()
