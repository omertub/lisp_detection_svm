# lisp_detection_svm

## pre_process:
python3 .\pre_process.py -h
usage: pre_process.py [-h] -w WORD [-ts TEST] [-tr TRAIN]

Preprocess wav files

optional arguments:
  -h, --help            show this help message and exit
  -w WORD, --word WORD  Word num to visualize
  -ts TEST, --test TEST
                        Generate test data
  -tr TRAIN, --train TRAIN
                        Generate train data
                        
example:
python3 .\visualization.py -w=0 -d='./' -s=1

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
python3 .\pre_process.py -ts=1 -w=0
