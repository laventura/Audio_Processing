import matplotlib.pyplot as plt
import numpy as np

import A4Part1, A4Part2, A4Part3

from loadTestCases import load



for ix in [1,2,3]:
    print(f"** Running test case {ix} **")
    test_case = load(3, ix)
    result = A4Part3.computeEngEnv(**test_case['input'])
    expected = test_case['output']
    assert result.shape == expected.shape, f"Test case {ix} failed. Expected {expected.shape}, got {result.shape}"
    assert np.allclose(result, expected), f"Test case {ix} fail: values not within tolerance"
    print(f"Test case {ix} passed\n")

