import unittest
from h2q_project.core import core_function  # 假设core.py中有core_function

class TestCore(unittest.TestCase):

    def test_core_function(self):
        # 编写你的测试用例
        self.assertEqual(core_function(2), 4) # 示例测试，假设core_function实现平方功能

if __name__ == '__main__':
    unittest.main()