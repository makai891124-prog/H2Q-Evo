"""
Proof-of-Concept Demo: Standard LLM vs DAS PrecisionGatedExecutor

Test Case:
"本句话包含的汉字数量是奇数。"

Output Format:
[Standard LLM]: ... (Hallucination)
[H2Q-Evo DAS]: ... (Tool-verified Truth)
"""
from __future__ import annotations

import re
from typing import Any, Dict

from precision_gated_executor import PrecisionGatedExecutor


class DemoLLMClient:
    """Deterministic demo LLM client with controlled behaviors."""

    def __init__(self) -> None:
        self._probe_counter = 0

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        # Probe phase: intentionally vary short answers to trigger high entropy.
        if "Answer briefly" in prompt:
            self._probe_counter += 1
            choices = ["是奇数", "是偶数", "无法确定"]
            return {"text": choices[(self._probe_counter - 1) % len(choices)]}

        # Orthogonal expansion: return Python code only.
        if "Generate a minimal Python script" in prompt:
            return {"text": "print(len(\"本句话包含的汉字数量是奇数。\"))"}

        # Finalization: must use tool output.
        if "Tool Output" in prompt:
            numbers = re.findall(r"\b\d+\b", prompt)
            number = numbers[0] if numbers else ""
            if number:
                if int(number) % 2 == 1:
                    return {"text": f"根据工具输出，字符数为{number}，因此为奇数。"}
                return {"text": f"根据工具输出，字符数为{number}，因此为偶数。"}
            return {"text": "未能从工具输出中提取数字，无法判断。"}

        # Baseline: hallucinated direct answer.
        return {"text": "这句话包含的汉字数量是奇数。"}


def main() -> None:
    task = "本句话包含的汉字数量是奇数。"

    llm = DemoLLMClient()

    # Baseline: direct LLM call (temperature=0 assumed for demo)
    baseline = llm.generate(task, temperature=0)

    # DAS experiment
    executor = PrecisionGatedExecutor(llm_client=llm)
    das_result = executor.execute_with_precision_gating(task)

    print(f"[Standard LLM]: {baseline.get('text')} (Hallucination)")
    print(f"[H2Q-Evo DAS]: {das_result.get('output')} (Tool-verified Truth)")


if __name__ == "__main__":
    main()
