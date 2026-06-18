from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Dict, Iterable, List


_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"


@dataclass
class PrimeStructure:
    value: int
    is_prime: bool
    factors: List[int]
    residue_mod_6: int
    binary: str


class PrimePromptEngineeringSystem:
    """Prompt-oriented prime analysis and encoding utility."""

    def build_analysis_prompt(self, value: int) -> str:
        return (
            "You are a mathematical structure analyzer. Determine if the number is prime, "
            "then provide factorization, modulo-6 residue, and binary structure. "
            f"Target number: {value}"
        )

    def is_prime(self, value: int) -> bool:
        if value < 2:
            return False
        if value in (2, 3):
            return True
        if value % 2 == 0 or value % 3 == 0:
            return False
        step = 5
        bound = isqrt(value)
        while step <= bound:
            if value % step == 0 or value % (step + 2) == 0:
                return False
            step += 6
        return True

    def factors(self, value: int) -> List[int]:
        if value < 2:
            return []
        out: List[int] = []
        remaining = value
        divisor = 2
        while divisor * divisor <= remaining:
            while remaining % divisor == 0:
                out.append(divisor)
                remaining //= divisor
            divisor += 1
        if remaining > 1:
            out.append(remaining)
        return out

    def analyze_prime_structure(self, values: Iterable[int]) -> Dict[str, List[Dict[str, object]]]:
        structures = []
        for value in values:
            if value < 0:
                raise ValueError("analyze_prime_structure expects non-negative integers")
            structure = PrimeStructure(
                value=value,
                is_prime=self.is_prime(value),
                factors=self.factors(value),
                residue_mod_6=value % 6,
                binary=bin(value)[2:],
            )
            structures.append(structure.__dict__)
        return {"structures": structures}

    def encode_primes(self, limit: int) -> Dict[str, object]:
        if limit < 2:
            return {"primes": [], "delta_base36": ""}

        primes = [n for n in range(2, limit + 1) if self.is_prime(n)]
        if not primes:
            return {"primes": [], "delta_base36": ""}

        deltas = [primes[0]] + [curr - prev for prev, curr in zip(primes, primes[1:])]
        encoded = ".".join(self._to_base36(delta) for delta in deltas)
        return {"primes": primes, "delta_base36": encoded}

    def decode_primes(self, encoded: str) -> List[int]:
        if not encoded:
            return []
        deltas = [self._from_base36(token) for token in encoded.split(".") if token]
        primes: List[int] = []
        running = 0
        for delta in deltas:
            running += delta
            primes.append(running)
        return primes

    def run_prime_coding_attempt(self, limit: int) -> Dict[str, object]:
        encoded = self.encode_primes(limit)
        analysis = self.analyze_prime_structure(encoded["primes"])
        return {
            "prompt_example": self.build_analysis_prompt(limit),
            "encoding": encoded,
            "analysis": analysis,
        }

    @staticmethod
    def _to_base36(value: int) -> str:
        if value == 0:
            return "0"
        if value < 0:
            raise ValueError("base36 encoder only accepts non-negative integers")

        chars: List[str] = []
        n = value
        while n > 0:
            n, rem = divmod(n, 36)
            chars.append(_BASE36[rem])
        return "".join(reversed(chars))

    @staticmethod
    def _from_base36(token: str) -> int:
        return int(token, 36)


__all__ = ["PrimePromptEngineeringSystem", "PrimeStructure"]
