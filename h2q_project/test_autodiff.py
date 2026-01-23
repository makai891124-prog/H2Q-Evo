import torch
import torch.autograd as autograd
import unittest
import h2q

class TestAutodiff(unittest.TestCase):

    def test_quaternion_addition_gradcheck(self):
        # Test gradient of quaternion addition
        q1 = h2q.random_quaternion(requires_grad=True)
        q2 = h2q.random_quaternion(requires_grad=True)
        torch.autograd.gradcheck(lambda x, y: x + y, (q1, q2))

    def test_quaternion_multiplication_gradcheck(self):
        # Test gradient of quaternion multiplication
        q1 = h2q.random_quaternion(requires_grad=True)
        q2 = h2q.random_quaternion(requires_grad=True)
        torch.autograd.gradcheck(lambda x, y: x * y, (q1, q2))

    def test_quaternion_conjugate_gradcheck(self):
        # Test gradient of quaternion conjugate
        q = h2q.random_quaternion(requires_grad=True)
        torch.autograd.gradcheck(lambda x: x.conj(), (q,))

    def test_quaternion_inverse_gradcheck(self):
        # Test gradient of quaternion inverse
        q = h2q.random_quaternion(requires_grad=True)
        # Need to avoid zero norm quaternions for inverse to be stable
        q = q + 1  # Ensure quaternion is not close to zero
        torch.autograd.gradcheck(lambda x: x.inverse(), (q,))

    def test_quaternion_exp_gradcheck(self):
        # Test gradient of quaternion exp
        q = h2q.random_quaternion(requires_grad=True)
        torch.autograd.gradcheck(lambda x: x.exp(), (q,))

    def test_quaternion_log_gradcheck(self):
        # Test gradient of quaternion log
        q = h2q.random_quaternion(requires_grad=True)
        # Need to avoid zero norm quaternions for log to be stable
        q = q + 1 # Ensure quaternion is not close to zero
        torch.autograd.gradcheck(lambda x: x.log(), (q,))

    def test_quaternion_normalize_gradcheck(self):
        # Test gradient of quaternion normalize
        q = h2q.random_quaternion(requires_grad=True)
        # Need to avoid zero norm quaternions for normalize to be stable
        q = q + 1 # Ensure quaternion is not close to zero
        torch.autograd.gradcheck(lambda x: x.normalize(), (q,))

    def test_quaternion_rotate_vector_gradcheck(self):
        # Test gradient of rotating a vector by a quaternion
        q = h2q.random_quaternion(requires_grad=True)
        v = torch.randn(3, requires_grad=True)
        torch.autograd.gradcheck(lambda quat, vec: quat.rotate(vec), (q, v))


if __name__ == '__main__':
    unittest.main()