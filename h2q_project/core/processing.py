import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(data):
    """Processes the input data and handles potential errors."""
    try:
        logging.debug("Starting data processing...")
        result = _perform_complex_calculation(data)
        logging.debug("Data processing completed successfully.")
        return result
    except ValueError as ve:
        logging.error(f"ValueError during data processing: {ve}")
        raise  # Re-raise the exception after logging
    except TypeError as te:
        logging.error(f"TypeError during data processing: {te}")
        raise
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        raise

def _perform_complex_calculation(data):
    """A placeholder for a complex calculation that might raise exceptions."""
    logging.debug("Performing complex calculation...")
    if not isinstance(data, int):
        raise TypeError("Input data must be an integer.")
    if data < 0:
        raise ValueError("Input data must be non-negative.")

    result = data * 2  # Example calculation
    logging.debug(f"Complex calculation result: {result}")
    return result
