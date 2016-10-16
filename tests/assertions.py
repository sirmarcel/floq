import unittest
import numpy as np

class CustomAssertions:
    def assertArrayEqual(self, a, b):
        if not np.array_equal(a,b):
            raise AssertionError('Arrays are not equal.')