# lisp_detection_svm

## pre_process:
python3 .\pre_process.py -h
usage: pre_process.py [-h] -w WORD [-t TEST] [-s SNR]

Preprocess wav files

optional arguments:
  -h, --help            show this help message and exit
  -w WORD, --word WORD  Word num to visualize
  -t TEST, --test TEST  Generate test data
  -s SNR, --snr SNR     Add noise with requested snr
                        
example:
python3 .\pre_process.py -w 0 -t 0 --snr 30

## visualization:

python3 .\visualization.py -h               
usage: visualization.py [-h] -w WORD [-m MFCC] [-n NORMALIZE] [-s SIGNAL] [-f FILE] [-d DIR]

Visualize wav files

optional arguments:
  -h, --help            show this help message and exit
  -w WORD, --word WORD  Word num to visualize
  -m MFCC, --mfcc MFCC  Visualize mfcc
  -n NORMALIZE, --normalize NORMALIZE
                        Normalize signals
  -s SIGNAL, --signal SIGNAL
                        Visualize signal
  -f FILE, --file FILE  Path to wav file
  -d DIR, --dir DIR     Path to directory containing wav files
  
exmaple:
python3 .\visualization.py -w=0 -d='./' -s=1
