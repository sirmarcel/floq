import unittest
import assertions
import numpy as np
import floq.fidelity as f
import floq.fixed_system as fs

u1 = np.array([[-0.1756 + 0.499j, -0.673 + 0.3477j, 0.0341 + 0.381j], [0.4613 + 0.3206j, 0.176 - 0.5432j, 0.0996 + 0.5903j], [0.5754 - 0.2709j, -0.2611 + 0.1789j, -0.7014 + 0.0582j]])
u2 = np.array([[0.2079 + 0.2374j, 0.5508 + 0.0617j, 0.3313 + 0.6953j], [-0.0423 - 0.2927j, 0.8017 - 0.1094j, -0.3873 - 0.3284j], [0.5487 + 0.7155j, 0.0209 - 0.1939j, -0.2633 - 0.2822j]])
u3 = np.array([[0.7025 + 0.2537j, -0.3209 + 0.1607j, 0.2748 - 0.4877j], [0.1706 - 0.1608j, 0.2006 - 0.9041j, -0.0438 - 0.2924j], [0.0347 + 0.6212j, 0.1164 + 0.0017j, -0.7627 - 0.1326j]])
u4 = np.array([[-0.2251 - 0.6249j, -0.5262 - 0.4341j, -0.0811 - 0.2947j], [0.3899 - 0.567j, 0.1173 + 0.5961j, -0.3893 - 0.0762j], [-0.1855 - 0.2256j, 0.0326 + 0.4055j, 0.8249 - 0.2623j]])

class TestTraceOverlap(unittest.TestCase,assertions.CustomAssertions):

    def test_overlap(self):
        system = fs.DummyFixedSystem(dim=3)
        overlap = f.trace_overlap(system,u1,u2)
        self.assertAlmostEqualWithDecimals(overlap,0.0378906,4)

class TestTraceOverlapDeriv(unittest.TestCase,assertions.CustomAssertions):

    def test_d_overlap(self):
        system = fs.DummyFixedSystem(dim=3)
        dus = np.array([u3,u4])
        target = np.array([0.302601,-0.291255])
        actual = f.d_trace_overlap(system,u1,dus,u2)
        self.assertArrayEqual(actual,target,4)

class TestHSProduct(unittest.TestCase,assertions.CustomAssertions):
    def test_product(self):
        product = f.hilbert_schmidt_product(u1,u2)
        self.assertAlmostEqualWithDecimals(product,0.113672 + 0.830189j,4)