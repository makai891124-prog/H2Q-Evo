import unittest
from h2q_project.self_reflection import SelfReflection

class TestSelfReflection(unittest.TestCase):

    def test_reflect_increment(self):
        reflection = SelfReflection(3)
        new_state, action = reflection.reflect()
        self.assertEqual(new_state, 4)
        self.assertEqual(action, "Incremented state")

    def test_reflect_decrement(self):
        reflection = SelfReflection(5)
        new_state, action = reflection.reflect()
        self.assertEqual(new_state, 4)
        self.assertEqual(action, "Decremented state")

    def test_initial_state(self):
        reflection = SelfReflection(10)
        self.assertEqual(reflection.state, 10)

if __name__ == '__main__':
    unittest.main()
