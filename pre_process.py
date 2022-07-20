import sys, os, argparse
from scipy.io.wavfile import read
from python_speech_features import mfcc
from sklearn.preprocessing import normalize
import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

# https://python-speech-features.readthedocs.io/en/latest/

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


def generate_features(wav_file):
    (frame_rate, samples) = read(wav_file)
    samples_clean = remove_noise(samples)
    samples_n = normalize_samples(samples_clean)
    cut = find_cut0(samples_n)
    samples_n = samples_n[cut:]
    mfcc_feat = mfcc(signal=samples_n, samplerate=frame_rate,
                     winlen=0.020, winstep=0.010, numcep=24, nfilt=40, nfft=1024, lowfreq=113, highfreq=6854)
    return mfcc_feat


def get_files_for_word(word):
    home_dir = "C:/Users/omer_/Desktop/recording_2/rec2"

    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(home_dir) for f in filenames if os.path.splitext(f)[1] == '.wav']
    files = [item.replace('\\','/') for item in result if word in item]
    return files


def extract_data_for_word(word, test):
    X, y = np.array([]), np.array([])

    files = get_files_for_word(word)

    length = 5
    first = 1
    length = len(files)
    for i in range(length):
        prec = (i/length) * 100
        print("%.2f" % prec, "%", end = '\r')
        mfcc_feat = generate_features(files[i])
        features = mfcc_feat.reshape(-1, 24)
        label = 1 if files[i][-6] == 'g' else 0

        ## single_word mode
        if test:
            X = features
            y = np.array(features.shape[0] * [label])
            y = y.reshape(-1,1)
            word_data = np.concatenate((X,y),axis=1)
            np.savetxt(f'test/t_{i}.csv', word_data, delimiter=",")

        ## multi_word mode
        else:
            if (first):
                X = features
                y = np.array(features.shape[0] * [label])
                first = 0
            else:
                X = np.concatenate((X, features))
                y = np.concatenate((y, features.shape[0] * [label]))

    if test:
        return 0
    y = y.reshape(-1,1)
    final_array = np.concatenate((X,y),axis=1)
    return final_array

WORDS = (
        "ספה",
        "פנס", 
        "קופסה", 
        "זחל",
        "רמז",
        "גזר",
    )

def main():
    # argparse
    parser = argparse.ArgumentParser(description='Preprocess wav files')
    parser.add_argument('-w', '--word', help='Word num to visualize', type=int, required=True)
    parser.add_argument('-ts', '--test', help='Generate test data')
    parser.add_argument('-tr', '--train', help='Generate train data')

    args = parser.parse_args()
    word = WORDS[args.word]

    if args.test:
        pass
    elif args.train:
        pass
    else:
        print("Please use test (-ts=1) or train (-tr=1)")
        exit(-1)

    print(word)
    word_data = extract_data_for_word(word, args.test)
    if args.train == 0:
        np.savetxt(f'./train/test_word.csv', word_data, delimiter=",")


if __name__ == '__main__':
    main()
