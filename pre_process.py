import random
import sys, os, argparse, math
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

# גזר, קופסה
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

def add_noise(samples):
    noisy_samples = np.zeros_like(samples)
    noise_factor=0.9
    (frame_rate, noise) = read("noise.wav")
    r = random.randint(0, len(noise) - len(samples))
    for i in range(len(samples)):
        noisy_samples[i]=noise_factor*samples[i]+(1-noise_factor)*noise[r+i]
    return noisy_samples

def add_noise_snr(samples, snr_db):
    (frame_rate, noise) = read("noise.wav")
    r = random.randint(0, len(noise) - len(samples))
    noise = noise[r:(r+len(samples))]

    snr = 10**(snr_db/10)
    Es = sum([(s**2) / len(samples) for s in samples])
    En = sum([(s**2) / len(noise) for s in noise])
    alpha = math.sqrt(Es/(snr*En))
    noisy_signal = samples + alpha*noise
    return noisy_signal


def generate_features(wav_file, word_data, test, snr):
    (frame_rate, samples) = read(wav_file)
    # if test and random.randint(0,1):
    #     samples = add_noise(samples)
    samples_clean = remove_noise(samples)
    samples_n = normalize_samples(samples_clean)
    if snr:
        samples_n = add_noise_snr(samples_n, snr)
    
    cut = word_data.cut_func(samples_n)

    if word_data.before:
        samples_n = samples_n[:cut]
    else:
        samples_n = samples_n[cut:]

    mfcc_feat = mfcc(signal=samples_n, samplerate=frame_rate,
                     winlen=0.020, winstep=0.010, numcep=24, nfilt=40, nfft=1024, lowfreq=113, highfreq=6854)

    return mfcc_feat


def get_files_for_word(word_data, test):
    if test:
        home_dir = "C:/Users/omer_/Desktop/recording_2/rec1"
    else:
        home_dir = "C:/Users/omer_/Desktop/recording_2/rec2"

    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(home_dir) for f in filenames if os.path.splitext(f)[1] == '.wav']
    files = [item.replace('\\','/') for item in result if (word_data.word in item)]
    return files


def extract_data_for_word(word_data, test, snr):
    X, y = np.array([]), np.array([])

    files = get_files_for_word(word_data, test)

    length = 5
    first = 1
    length = len(files)
    for i in range(length):
        prec = (i/length) * 100
        print("%.2f" % prec, "%", end = '\r')
        mfcc_feat = generate_features(files[i], word_data, test, snr)
        features = mfcc_feat.reshape(-1, 24)
        label = 1 if files[i][-6] == 'g' else 0
        
        # use for labels generation
        # if (first):
        #     final_array = np.array([label])
        #     first = 0
        # else:
        #     final_array = np.concatenate((final_array, [label]))
        # continue

        ## single_word mode
        if test:
            X = features
            y = np.array(features.shape[0] * [label])
            y = y.reshape(-1,1)
            word_info = np.concatenate((X,y),axis=1)
            np.savetxt(f'test/t_{i}.csv', word_info, delimiter=",")

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


class word_data_t:
    def __init__(self, word, before, cut_func):
        self.word = word
        self.before = before
        self.cut_func = cut_func


# creating WORDS       
WORDS = [] 
  
# appending instances to WORDS 
WORDS.append( word_data_t("ספה",   True,   find_cut0) )
WORDS.append( word_data_t("פנס",   False,  find_cut5) )
WORDS.append( word_data_t("קופסה", False,  find_cut5) )
WORDS.append( word_data_t("זחל",   True,   find_cut3) )
WORDS.append( word_data_t("רמז",   False,  find_cut4) )
WORDS.append( word_data_t("גזר",   False,  find_cut5) )

def main():
    # argparse
    parser = argparse.ArgumentParser(description='Preprocess wav files')
    parser.add_argument('-w', '--word', help='Word num to visualize', type=int, required=True)
    parser.add_argument('-t', '--test', help='Generate test data', type=int)
    parser.add_argument('-s', '--snr', help='Add noise with requested snr')


    args = parser.parse_args()
    word_data = WORDS[args.word]

    test = bool(args.test)
    if test:
        print("generating test")
    else:
        print("generating train")

    word_info = extract_data_for_word(word_data, test, args.snr)
    if not test:
        np.savetxt(f'train/test_word.csv', word_info, delimiter=",")


if __name__ == '__main__':
    main()
