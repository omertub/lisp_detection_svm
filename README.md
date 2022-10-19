# lisp_detection_svm

## pre_process:
python3 .\pre_process.py -h
usage: pre_process.py [-h] -w WORD [-t TEST] [-s SNR]  

Preprocess wav files  

optional arguments:  
&ensp;-h, --help&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;show this help message and exit  
&ensp;-w WORD, --word WORD&ensp;&nbsp;Word num to pre-process  
&ensp;-t TEST, --test TEST&emsp;&emsp;&emsp;&nbsp;Generate test data  
&ensp;-s SNR, --snr SNR&emsp;&emsp;&emsp;&emsp;Add noise with requested snr  
                        
example:  
python3 .\pre_process.py -w 0 -t 0 --snr 30  

## visualization:

python3 .\visualization.py -h               
usage: visualization.py [-h] -w WORD [-m MFCC] [-n NORMALIZE] [-s SIGNAL] [-f FILE] [-d DIR]  

Visualize wav files  

optional arguments:  
&ensp;-h, --help&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;show this help message and exit  
&ensp;-w WORD, --word WORD&ensp;&nbsp;Word num to visualize  
&ensp;-m MFCC, --mfcc MFCC&emsp;&ensp;Visualize mfcc  
&ensp;-n NORMALIZE, --normalize NORMALIZE  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Normalize signals  
&ensp;-s SIGNAL, --signal SIGNAL  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Visualize signal  
&ensp;-f FILE, --file FILE&emsp;&emsp;&emsp;&emsp;&nbsp;Path to wav file  
&ensp;-d DIR, --dir DIR&emsp;&emsp;&emsp;&emsp;&emsp;Path to directory containing wav files  
  
exmaple:  
python3 .\visualization.py -w=0 -d='./' -s=1  
