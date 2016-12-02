from unittest import TestCase
import numpy as np
import floq.helpers.blockmatrix as bm


class TestGetBlockFromMatrix(TestCase):
    def setUp(self):
        self.dim_block = 5
        self.n_block = 3

        self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h, self.i  \
            = [j*np.ones([self.dim_block, self.dim_block]) for j in range(9)]

        matrix = np.bmat([[self.a, self.b, self.c],
                         [self.d, self.e, self.f],
                         [self.g, self.h, self.i]])
        self.matrix = np.array(matrix)

    def test_a(self):
        block = bm.get_block_from_matrix(self.matrix, self.dim_block, self.n_block, 0, 0)
        self.assertTrue(np.array_equal(block, self.a))

    def test_b(self):
        block = bm.get_block_from_matrix(self.matrix, self.dim_block, self.n_block, 0, 1)
        self.assertTrue(np.array_equal(block, self.b))

    def test_c(self):
        block = bm.get_block_from_matrix(self.matrix, self.dim_block, self.n_block, 0, 2)
        self.assertTrue(np.array_equal(block, self.c))

    def test_d(self):
        block = bm.get_block_from_matrix(self.matrix, self.dim_block, self.n_block, 1, 0)
        self.assertTrue(np.array_equal(block, self.d))

    def test_e(self):
        block = bm.get_block_from_matrix(self.matrix, self.dim_block, self.n_block, 1, 1)
        self.assertTrue(np.array_equal(block, self.e))

    def test_f(self):
        block = bm.get_block_from_matrix(self.matrix, self.dim_block, self.n_block, 1, 2)
        self.assertTrue(np.array_equal(block, self.f))

    def test_g(self):
        block = bm.get_block_from_matrix(self.matrix, self.dim_block, self.n_block, 2, 0)
        self.assertTrue(np.array_equal(block, self.g))

    def test_h(self):
        block = bm.get_block_from_matrix(self.matrix, self.dim_block, self.n_block, 2, 1)
        self.assertTrue(np.array_equal(block, self.h))

    def test_i(self):
        block = bm.get_block_from_matrix(self.matrix, self.dim_block, self.n_block, 2, 2)
        self.assertTrue(np.array_equal(block, self.i))


class TestSetBlockInMatrix(TestCase):
    def setUp(self):
        self.dim_block = 5
        self.n_block = 3

        self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h, self.i \
            = [j*np.ones([self.dim_block, self.dim_block]) for j in range(9)]

        matrix = np.bmat([[self.a, self.b, self.c],
                         [self.d, self.e, self.f],
                         [self.g, self.h, self.i]])
        self.original = np.array(matrix)

        total_size = self.dim_block*self.n_block
        self.copy = np.zeros([total_size,total_size])

    def test_set(self):
        # Try to recreate self.original with the new function
        bm.set_block_in_matrix(self.a, self.copy, self.dim_block, self.n_block, 0, 0)
        bm.set_block_in_matrix(self.b, self.copy, self.dim_block, self.n_block, 0, 1)
        bm.set_block_in_matrix(self.c, self.copy, self.dim_block, self.n_block, 0, 2)
        bm.set_block_in_matrix(self.d, self.copy, self.dim_block, self.n_block, 1, 0)
        bm.set_block_in_matrix(self.e, self.copy, self.dim_block, self.n_block, 1, 1)
        bm.set_block_in_matrix(self.f, self.copy, self.dim_block, self.n_block, 1, 2)
        bm.set_block_in_matrix(self.g, self.copy, self.dim_block, self.n_block, 2, 0)
        bm.set_block_in_matrix(self.h, self.copy, self.dim_block, self.n_block, 2, 1)
        bm.set_block_in_matrix(self.i, self.copy, self.dim_block, self.n_block, 2, 2)
       
        self.assertTrue(np.array_equal(self.copy,self.original))

