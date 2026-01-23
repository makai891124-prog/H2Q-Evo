import unittest
from h2q_project import utils


class TestUtils(unittest.TestCase):
    def test_add(self):
        self.assertEqual(utils.add(1, 2), 3)
        self.assertEqual(utils.add(-1, 1), 0)
        self.assertEqual(utils.add(0, 0), 0)

    def test_subtract(self):
        self.assertEqual(utils.subtract(5, 2), 3)
        self.assertEqual(utils.subtract(1, 1), 0)
        self.assertEqual(utils.subtract(0, 5), -5)


if __name__ == '__main__':
    unittest.main()