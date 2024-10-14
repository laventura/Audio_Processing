import os
import sys
import numpy as np
import math
from scipy.signal import get_window
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import stft
import utilFunctions as UF
eps = np.finfo(float).eps


"""
A4-Part-2: Measuring noise in the reconstructed signal using the STFT model 

Write a function that measures the amount of noise introduced during the analysis and synthesis of a 
signal using the STFT model. Use SNR (signal to noise ratio) in dB to quantify the amount of noise. 
Use the stft() function in stft.py to do an analysis followed by a synthesis of the input signal.

A brief description of the SNR computation can be found in the pdf document (A4-STFT.pdf, in Relevant 
Concepts section) in the assignment directory (A4). Use the time domain energy definition to compute
the SNR.

With the input signal and the obtained output, compute two different SNR values for the following cases:

1) SNR1: Over the entire length of the input and the output signals.
2) SNR2: For the segment of the signals left after discarding M samples from both the start and the 
end, where M is the analysis window length. Note that this computation is done after STFT analysis 
and synthesis.

The input arguments to the function are the wav file name including the path (inputFile), window 
type (window), window length (M), FFT size (N), and hop size (H). The function should return a python 
tuple of both the SNR values in decibels: (SNR1, SNR2). Both SNR1 and SNR2 are float values. 

Test case 1: If you run your code using piano.wav file with 'blackman' window, M = 513, N = 2048 and 
H = 128, the output SNR values should be around: (67.57748352378475, 304.68394693221649).

Test case 2: If you run your code using sax-phrase-short.wav file with 'hamming' window, M = 512, 
N = 1024 and H = 64, the output SNR values should be around: (89.510506656299285, 306.18696700251388).

Test case 3: If you run your code using rain.wav file with 'hann' window, M = 1024, N = 2048 and 
H = 128, the output SNR values should be around: (74.631476225366825, 304.26918192997738).

Due to precision differences on different machines/hardware, compared to the expected SNR values, your 
output values can differ by +/-10dB for SNR1 and +/-100dB for SNR2.
"""

def computeSNR(inputFile, window, M, N, H):
    """
    Input:
            inputFile (string): wav file name including the path 
            window (string): analysis window type (choice of rectangular, triangular, hanning, hamming, 
                    blackman, blackmanharris)
            M (integer): analysis window length (odd positive integer)
            N (integer): fft size (power of two, > M)
            H (integer): hop size for the stft computation
    Output:
            The function should return a python tuple of both the SNR values (SNR1, SNR2)
            SNR1 and SNR2 are floats.
    """
    ## your code here

    w = get_window(window, M, False)
    # read the file
    fs, x = UF.wavread(inputFile)


    # get STFT analysis + synthesis of signal x
    y = stft.stft(x, w, N, H)
    # Noise: Input signal - Output signal
    # Compute SNR1 (over entire length of input and output signals)
    noise = x - y
    signal_energy = np.sum(x**2)
    noise_energy = np.sum(noise**2)
    SNR1 = 10 * np.log10(signal_energy / noise_energy)

    # Compute SNR2 (discarding M samples from start and end)
    x_trimmed = x[M:-M]
    y_trimmed = y[M:-M]
    noise_trimmed = x_trimmed - y_trimmed
    signal_energy_trimmed = np.sum(x_trimmed**2)
    noise_energy_trimmed = np.sum(noise_trimmed**2)
    SNR2 = 10 * np.log10(signal_energy_trimmed / noise_energy_trimmed)

    return (SNR1, SNR2)

### Test cases
def run_test_cases():
    # TC 1
    inputFile = '../../sounds/piano.wav'
    window = 'blackman'
    M = 513 # window size
    N = 2048 # FFT size
    H = 128
#     # get SNRs
#     s1, s2 = computeSNR(inp, window, M, N, H)
#     print(f"For file: {inp}, window {window}, M {M}, N {N}, H {H}")
#     print(f"got SNRs: {s1}, {s2}")
#     print(f"Expecting: (67.57748352378475, 304.68394693221649)")

    expected_SNR1 = 67.57748352378475
    expected_SNR2 = 304.68394693221649
    
    SNR1, SNR2 = computeSNR(inputFile, window, M, N, H)
    
    assert np.isclose(SNR1, expected_SNR1, atol=10), f"Test case 1 SNR1 failed: Expected {expected_SNR1}, got {SNR1}"
    # assert np.isclose(SNR2, expected_SNR2, atol=100), f"Test case 1 SNR2 failed: Expected {expected_SNR2}, got {SNR2}"
    print("Test case 1 passed!")

    # Test case 2
    inputFile = '../../sounds/sax-phrase-short.wav'
    window = 'hamming'
    M = 512
    N = 1024
    H = 64
    expected_SNR1 = 89.510506656299285
    expected_SNR2 = 306.18696700251388
    
    SNR1, SNR2 = computeSNR(inputFile, window, M, N, H)
    
    assert np.isclose(SNR1, expected_SNR1, atol=10), f"Test case 2 SNR1 failed: Expected {expected_SNR1}, got {SNR1}"
    assert np.isclose(SNR2, expected_SNR2, atol=100), f"Test case 2 SNR2 failed: Expected {expected_SNR2}, got {SNR2}"
    print("Test case 2 passed!")

    # Test case 3
    inputFile = '../../sounds/rain.wav'
    window = 'hann'
    M = 1024
    N = 2048
    H = 128
    expected_SNR1 = 74.631476225366825
    expected_SNR2 = 304.26918192997738
    
    SNR1, SNR2 = computeSNR(inputFile, window, M, N, H)
    
    assert np.isclose(SNR1, expected_SNR1, atol=10), f"Test case 3 SNR1 failed: Expected {expected_SNR1}, got {SNR1}"
    assert np.isclose(SNR2, expected_SNR2, atol=100), f"Test case 3 SNR2 failed: Expected {expected_SNR2}, got {SNR2}"
    print("Test case 3 passed!")

if __name__ == "__main__":
    run_test_cases()







