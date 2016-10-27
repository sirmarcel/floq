import unittest
import numpy as np

class CustomAssertions:
    def assertArrayEqual(self, a, b, decimals = 5):
        if not np.allclose(a,b,atol=0.1**decimals):
            print a.round(decimals)
            print b.round(decimals)
            raise AssertionError('Arrays are not equal!')