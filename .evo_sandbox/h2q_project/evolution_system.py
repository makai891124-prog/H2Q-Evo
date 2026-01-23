import os
import openai

MODEL_NAME = "gpt-3.5-turbo"

def select_model():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "gpt-3.5-turbo"  # Default model if no API key

    try:
        # Add logic to determine the best model based on API key capabilities
        # This is a placeholder; replace with actual API call and model selection logic
        # Example: Check if the API key has access to gpt-4
        openai.api_key = api_key
        models = openai.Model.list()
        model_names = [model.id for model in models['data']]
        if 'gpt-4' in model_names:
            return "gpt-4"  # Use gpt-4 if available
        else:
            return "gpt-3.5-turbo"
    except Exception as e:
        print(f"Error selecting model: {e}")
        return "gpt-3.5-turbo"  # Fallback to default model on error


MODEL_NAME = select_model()


def generate_improvement_prompt(current_code, task_description):
    return f"Improve the following code:\n{current_code}\nTask: {task_description}"


def evolve_code(current_code, task_description):
    prompt = generate_improvement_prompt(current_code, task_description)

    openai.api_key = os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    improved_code = response.choices[0].message['content']
    return improved_code