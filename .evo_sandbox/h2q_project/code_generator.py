import json

class CodeGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate_code(self, task_description, context=None):
        """Generates code based on the task description and context.

        Args:
            task_description (str): A description of the task to be performed.
            context (str, optional): Additional context information. Defaults to None.

        Returns:
            str: The generated code, or None if generation fails.
        """
        try:
            prompt = self._construct_prompt(task_description, context)
            code = self.llm_client.generate(prompt)

            # Basic JSON format validation - crucial for reliability
            try:
                json.loads(code)
            except json.JSONDecodeError:
                print("Error: Code generation failed - Invalid JSON format.")
                return None

            # Enhanced error handling and logging could be added here
            # Example: log the prompt and the raw output from the LLM for debugging

            return code
        except Exception as e:
            print(f"Error during code generation: {e}")
            return None

    def _construct_prompt(self, task_description, context=None):
        """Constructs the prompt to be sent to the LLM.

        Args:
            task_description (str): The description of the task.
            context (str, optional): Additional context information. Defaults to None.

        Returns:
            str: The constructed prompt.
        """

        prompt = f"You are a helpful AI assistant that generates JSON code snippets. " \
                 f"The JSON should be valid and directly executable. " \
                 f"Your response should contain only the JSON object, nothing else. " \
                 f"Task: {task_description}"

        if context:
            prompt += f"\nContext: {context}"

        prompt += f"\nEnsure the output is a valid JSON object."

        return prompt
