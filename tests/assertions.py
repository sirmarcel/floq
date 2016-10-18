import unittest
import numpy as np

class CustomAssertions:
    def assertArrayEqual(self, a, b, decimals = 5):
        if not np.array_equal(a.round(decimals),b.round(decimals)):
            print a.round(decimals)
            print b.round(decimals)
            raise AssertionError('Arrays are not equal!')