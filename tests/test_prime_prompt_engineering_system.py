import unittest

from h2q_project.services.prime_prompt_engineering_system import PrimePromptEngineeringSystem


class PrimePromptEngineeringSystemTests(unittest.TestCase):
    def setUp(self):
        self.system = PrimePromptEngineeringSystem()

    def test_prime_judgement(self):
        self.assertTrue(self.system.is_prime(2))
        self.assertTrue(self.system.is_prime(29))
        self.assertFalse(self.system.is_prime(1))
        self.assertFalse(self.system.is_prime(35))

    def test_structure_analysis(self):
        result = self.system.analyze_prime_structure([2, 4, 29])
        structures = result["structures"]

        self.assertEqual(structures[0]["is_prime"], True)
        self.assertEqual(structures[1]["factors"], [2, 2])
        self.assertEqual(structures[2]["residue_mod_6"], 5)
        self.assertEqual(structures[2]["binary"], "11101")

    def test_prime_encoding_roundtrip(self):
        encoded = self.system.encode_primes(50)
        decoded = self.system.decode_primes(encoded["delta_base36"])

        self.assertEqual(decoded, encoded["primes"])
        self.assertIn(47, decoded)


if __name__ == "__main__":
    unittest.main()
