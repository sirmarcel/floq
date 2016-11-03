import unittest
import assertions
import numpy as np
import floq.fidelity as f
import floq.fixed_system as fs

class TestHSProduct(unittest.TestCase,assertions.CustomAssertions):
    def test_product(self):
        u1 = np.array([[-0.1756 + 0.499j, -0.673 + 0.3477j, 0.0341 + 0.381j], [0.4613 + 0.3206j, 0.176 - 0.5432j, 0.0996 + 0.5903j], [0.5754 - 0.2709j, -0.2611 + 0.1789j, -0.7014 + 0.0582j]])
        u2 = np.array([[0.2079 + 0.2374j, 0.5508 + 0.0617j, 0.3313 + 0.6953j], [-0.0423 - 0.2927j, 0.8017 - 0.1094j, -0.3873 - 0.3284j], [0.5487 + 0.7155j, 0.0209 - 0.1939j, -0.2633 - 0.2822j]])
        product = f.hilbert_schmidt_product(u1,u2)
        
        self.assertAlmostEqualWithDecimals(product,0.113672 + 0.830189j,4)

class TestTraceOverlap(unittest.TestCase,assertions.CustomAssertions):

    def test_overlap(self):
        u1 = np.array([[-0.1756 + 0.499j, -0.673 + 0.3477j, 0.0341 + 0.381j], [0.4613 + 0.3206j, 0.176 - 0.5432j, 0.0996 + 0.5903j], [0.5754 - 0.2709j, -0.2611 + 0.1789j, -0.7014 + 0.0582j]])
        u2 = np.array([[0.2079 + 0.2374j, 0.5508 + 0.0617j, 0.3313 + 0.6953j], [-0.0423 - 0.2927j, 0.8017 - 0.1094j, -0.3873 - 0.3284j], [0.5487 + 0.7155j, 0.0209 - 0.1939j, -0.2633 - 0.2822j]])
        system = fs.DummyFixedSystem(dim=3)
        product = f.trace_overlap(system,u1,u2)
        
        self.assertAlmostEqualWithDecimals(product,0.0378906,4)