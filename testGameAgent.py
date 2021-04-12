import unittest

import numpy as np

from lp import GameAgent


class TestRPS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = GameAgent()

    def test_case_1(self):
        R = [
            [0, 1, -1], [-1, 0, 1], [1, -1, 0]
        ]

        np.testing.assert_almost_equal(
            self.agent.solve(R),
            np.array([0.333, 0.333, 0.333]),
            decimal=3
        )

    def test_case_2(self):
        R = [[0, 2, -1],
             [-2, 0, 1],
             [1, -1, 0]]

        np.testing.assert_almost_equal(
            self.agent.solve(R),
            np.array([0.250, 0.250, 0.500]),
            decimal=3
        )


unittest.main(argv=[''], verbosity=2, exit=False)
