import logging

# Configure logging (if not already configured elsewhere)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_operation(func, *args, **kwargs):
    """Executes a function and logs any exceptions that occur."""
    try:
        logging.debug(f"Executing function: {func.__name__} with args: {args} and kwargs: {kwargs}")
        result = func(*args, **kwargs)
        logging.debug(f"Function {func.__name__} executed successfully.")
        return result
    except Exception as e:
        logging.exception(f"Error occurred during function {func.__name__} execution: {e}")
        return None # or raise, depending on the desired behavior
