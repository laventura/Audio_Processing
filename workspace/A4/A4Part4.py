import os
import sys
import numpy as np
from scipy.signal import get_window
import matplotlib.pyplot as plt
import math

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import stft
import utilFunctions as UF

eps = np.finfo(float).eps

"""
A4-Part-4: Computing onset detection function (Optional)

Write a function to compute a simple onset detection function (ODF) using the STFT. Compute two ODFs 
one for each of the frequency bands, low and high. The low frequency band is the set of all the 
frequencies between 0 and 3000 Hz and the high frequency band is the set of all the frequencies 
between 3000 and 10000 Hz (excluding the boundary frequencies in both the cases). 

A brief description of the onset detection function can be found in the pdf document (A4-STFT.pdf, 
in Relevant Concepts section) in the assignment directory (A4). Start with an initial condition of 
ODF(0) = 0 in order to make the length of the ODF same as that of the energy envelope. Remember to 
apply a half wave rectification on the ODF. 

The input arguments to the function are the wav file name including the path (inputFile), window 
type (window), window length (M), FFT size (N), and hop size (H). The function should return a numpy 
array with two columns, where the first column is the ODF computed on the low frequency band and the 
second column is the ODF computed on the high frequency band.

Use stft.stftAnal() to obtain the STFT magnitude spectrum for all the audio frames. Then compute two 
energy values for each frequency band specified. While calculating frequency bins for each frequency 
band, consider only the bins that are within the specified frequency range. For example, for the low 
frequency band consider only the bins with frequency > 0 Hz and < 3000 Hz (you can use np.where() to 
find those bin indexes). This way we also remove the DC offset in the signal in energy envelope 
computation. The frequency corresponding to the bin index k can be computed as k*fs/N, where fs is 
the sampling rate of the signal.

To get a better understanding of the energy envelope and its characteristics you can plot the envelopes 
together with the spectrogram of the signal. You can use matplotlib plotting library for this purpose. 
To visualize the spectrogram of a signal, a good option is to use colormesh. You can reuse the code in
sms-tools/lectures/4-STFT/plots-code/spectrogram.py. Either overlay the envelopes on the spectrogram 
or plot them in a different subplot. Make sure you use the same range of the x-axis for both the 
spectrogram and the energy envelopes.

NOTE: Running these test cases might take a few seconds depending on your hardware.

Test case 1: Use piano.wav file with window = 'blackman', M = 513, N = 1024 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 69 (69 samples) and of the high frequency 
band span from 70 to 232 (163 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 2: Use piano.wav file with window = 'blackman', M = 2047, N = 4096 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 278 (278 samples) and of the high frequency 
band span from 279 to 928 (650 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 3: Use sax-phrase-short.wav file with window = 'hamming', M = 513, N = 2048 and H = 256 as 
input. The bin indexes of the low frequency band span from 1 to 139 (139 samples) and of the high 
frequency band span from 140 to 464 (325 samples). To numerically compare your output, use 
loadTestCases.py script to obtain the expected output.

In addition to comparing results with the expected output, you can also plot your output for these 
test cases. For test case 1, you can clearly see that the ODFs have sharp peaks at the onset of the 
piano notes (See figure in the accompanying pdf). You will notice exactly 6 peaks that are above 
10 dB value in the ODF computed on the high frequency band. 
"""

def computeODF(inputFile, window, M, N, H):
    """
    Inputs:
            inputFile (string): input sound file (monophonic with sampling rate of 44100)
            window (string): analysis window type (choice of rectangular, triangular, hanning, hamming, 
                blackman, blackmanharris)
            M (integer): analysis window size (odd integer value)
            N (integer): fft size (power of two, bigger or equal than than M)
            H (integer): hop size for the STFT computation
    Output:
            The function should return a numpy array with two columns, where the first column is the ODF 
            computed on the low frequency band and the second column is the ODF computed on the high 
            frequency band.
            ODF[:,0]: ODF computed in band 0 < f < 3000 Hz 
            ODF[:,1]: ODF computed in band 3000 < f < 10000 Hz
    """
    
    ### your code here
    # read the audio file
    fs, x = UF.wavread(inputFile)

    # compute STFT of the signal
    w = get_window(window, M)
    mX, pX = stft.stftAnal(x, w, N, H)

    # compute frequency bins
    fbin = np.arange(N/2 + 1)*float(fs)/N 

    # find bin indices for the two freq bands: Low and High
    low_band = np.where((fbin > 0) & (fbin < 3000))[0]
    hi_band  = np.where((fbin > 3000) & (fbin <10000))[0]
    print(f"[debug: num low_band bins: {len(low_band)}, high_band bins: {len(hi_band)}]")

    # init ODF array
    ODF = np.zeros((mX.shape[0], 2)) 

    ## compute energey for each frame and band
    # convert from dB to linear for summation
    E_low  = np.sum(10**(mX[:, low_band]/20), axis=1)
    E_high = np.sum(10**(mX[:, hi_band]/20), axis=1)

    # compute ODF for low-freq band
    for i in range(1, len(E_low)):
        ODF[i, 0] = max(0, E_low[i] - E_low[i-1])
    # compute ODF for high-freq band
    for i in range(1, len(E_high)):
        ODF[i, 1] = max(0, E_high[i] - E_high[i-1])
    
    # convert to dB scale
    ODF = 10 * np.log10(ODF + eps)

    return ODF

## test
def plot_spectrogram_and_ODFs(inputFile, window, M, N, H):
    fs, x = UF.wavread(inputFile)
    w = get_window(window, M)
    mX, pX = stft.stftAnal(x, w, N, H)
    ODF = computeODF(inputFile, window, M, N, H)

    plt.figure(figsize=(12, 8))

    # Plot spectrogram
    plt.subplot(211)
    numFrames = int(mX[:,0].size)
    frmTime = H*np.arange(numFrames)/float(fs)
    binFreq = fs*np.arange(N/2 +1)/float(N)
    plt.pcolormesh(frmTime, binFreq, np.transpose(mX))
    plt.title(f"Spectrogram {inputFile}")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.autoscale(tight=True)

    # Plot ODFs
    plt.subplot(212)
    plt.plot(frmTime, ODF[:,0], label='Low Band (0-3000 Hz)')
    plt.plot(frmTime, ODF[:,1], label='High Band (3000-10000 Hz)')
    plt.title('Onset Detection Functions')
    plt.xlabel('Time (s)')
    plt.ylabel('ODF (dB)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Test case 1
    audioFile = '../../sounds/piano.wav'  # sax-phrase-short.wav
    M = 513
    N = 1024
    window = 'blackman'
    H = 128
    print(f"[Test 1] file: {audioFile}, window {window}, M {M}, N {N}, H {H}")
    plot_spectrogram_and_ODFs(audioFile, window, M, N, H)
