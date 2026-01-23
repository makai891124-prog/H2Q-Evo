import random
import string
from h2q_project.utils.logger import setup_logger, log_event


def generate_code():
    """Generates a random code snippet."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=20))


def validate_code(code):
    """Validates the generated code (dummy implementation)."""
    return "Valid" if random.random() > 0.5 else "Invalid"


def run_evolution(task_id):
    """Simulates an evolutionary process with logging."""
    logger = setup_logger(task_id)
    log_event(logger, 'info', f'Task {task_id} started.')

    try:
        code = generate_code()
        log_event(logger, 'info', f'Code generated: {code}')

        validation_result = validate_code(code)
        log_event(logger, 'info', f'Code validation result: {validation_result}')

        if validation_result == "Valid":
            log_event(logger, 'info', f'Task {task_id} completed successfully.')
        else:
            log_event(logger, 'error', f'Task {task_id} failed due to invalid code.')

    except Exception as e:
        log_event(logger, 'error', f'Task {task_id} failed with exception: {e}')

    finally:
        pass

if __name__ == '__main__':
    task_id = 'evolution_task_1'
    run_evolution(task_id)
