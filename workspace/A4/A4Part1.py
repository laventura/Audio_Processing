import numpy as np
from scipy.signal import get_window
from scipy.fftpack import fft, fftshift
import math
import matplotlib.pyplot as plt
eps = np.finfo(float).eps

""" 
A4-Part-1: Extracting the main lobe of the spectrum of a window

Write a function that extracts the main lobe of the magnitude spectrum of a window given a window 
type and its length (M). The function should return the samples corresponding to the main lobe in 
decibels (dB).

To compute the spectrum, take the FFT size (N) to be 8 times the window length (N = 8*M) (For this 
part, N need not be a power of 2). 

The input arguments to the function are the window type (window) and the length of the window (M). 
The function should return a numpy array containing the samples corresponding to the main lobe of 
the window. 

In the returned numpy array you should include the samples corresponding to both the local minimas
across the main lobe. 

The possible window types that you can expect as input are rectangular ('boxcar'), 'hamming' or
'blackmanharris'.

NOTE: You can approach this question in two ways: 1) You can write code to find the indices of the 
local minimas across the main lobe. 2) You can manually note down the indices of these local minimas 
by plotting and a visual inspection of the spectrum of the window. If done manually, the indices 
have to be obtained for each possible window types separately (as they differ across different 
window types).

Tip: log10(0) is not well defined, so its a common practice to add a small value such as eps = 1e-16 
to the magnitude spectrum before computing it in dB. This is optional and will not affect your answers. 
If you find it difficult to concatenate the two halves of the main lobe, you can first center the 
spectrum using fftshift() and then compute the indexes of the minimas around the main lobe.


Test case 1: If you run your code using window = 'blackmanharris' and M = 100, the output numpy 
array should contain 65 samples.

Test case 2: If you run your code using window = 'boxcar' and M = 120, the output numpy array 
should contain 17 samples.

Test case 3: If you run your code using window = 'hamming' and M = 256, the output numpy array 
should contain 33 samples.

"""
def extractMainLobe(window, M):
    """
    Input:
            window (string): Window type to be used (Either rectangular ('boxcar'), 'hamming' or '
                blackmanharris')
            M (integer): length of the window to be used
    Output:
            The function should return a numpy array containing the main lobe of the magnitude 
            spectrum of the window in decibels (dB).
    """

    w = get_window(window, M)         # get the window 
    
    ### Your code here
    N = 8 * M # FFT size
    # compute the FFT
    X = fft(w, N)
    # compute magnitude spectrum in dB
    mX = 20 * np.log10(abs(X) + eps)
    # shift spectrum so the center freq is at the middle
    mX = fftshift(mX)
    
    # find the peak main lobe
    peak_index = np.argmax(mX)

    # find the local minima on each side of the center
    left_ix = peak_index
    while left_ix > 0 and mX[left_ix - 1] <= mX[left_ix]:
        left_ix -= 1
    
    right_ix = peak_index
    while right_ix < len(mX)-1 and mX[right_ix+1] <= mX[right_ix]:
        right_ix += 1
    
    main_lobe = mX[left_ix: right_ix+1]

    return main_lobe

### Test cases
def test_extractMainLobe():
    # Test case 1
    window = 'blackmanharris'
    M = 100
    print(f"Running test case: {window}")
    result = extractMainLobe(window, M)
    assert len(result) == 65, f"Test case 1 failed: Expected 65 samples, got {len(result)}"
    print("Test case 1 passed!")

    # Test case 2
    window = 'boxcar'
    M = 120
    print(f"Running test case: {window}")
    result = extractMainLobe(window, M)
    assert len(result) == 17, f"Test case 2 failed: Expected 17 samples, got {len(result)}"
    print("Test case 2 passed!")

    # Test case 3
    window = 'hamming'
    M = 256
    print(f"Running test case: {window}")
    result = extractMainLobe(window, M)
    assert len(result) == 33, f"Test case 3 failed: Expected 33 samples, got {len(result)}"
    print("Test case 3 passed!")

    print("All test cases passed successfully!")

if __name__ == "__main__":
    test_extractMainLobe()