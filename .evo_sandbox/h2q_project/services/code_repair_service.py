from h2q_project.services.code_generation_service import CodeGenerationService


class CodeRepairService:
    def __init__(self, code_generation_service: CodeGenerationService):
        self.code_generation_service = code_generation_service

    def repair_code(self, code: str, error_message: str, task_description: str) -> str:
        """Repairs the given code based on the error message and task description.

        Args:
            code: The code to repair.
            error_message: The error message received during execution.
            task_description: The original task description.

        Returns:
            The repaired code.
        """
        prompt = f"""You are an expert code repairer.  You will be given a piece of code, an error message, and the original task description that the code was intended to solve.  Your job is to repair the code so that it correctly implements the task and does not produce the error.  Return only the corrected code.  Do not include any explanations.

Original Task: {task_description}

Original Code:
{code}

Error Message:
{error_message}

Corrected Code:
"""

        repaired_code = self.code_generation_service.generate_code(prompt)

        return repaired_code
