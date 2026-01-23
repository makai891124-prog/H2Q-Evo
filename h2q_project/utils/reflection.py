import inspect

def get_class_from_string(module_name, class_name):
    """Dynamically imports a class from a string.

    Args:
        module_name (str): The name of the module.
        class_name (str): The name of the class.

    Returns:
        The class object, or None if not found.
    """
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        return None


def get_function_from_string(module_name, function_name):
    """Dynamically imports a function from a string.

    Args:
        module_name (str): The name of the module.
        function_name (str): The name of the function.

    Returns:
        The function object, or None if not found.
    """
    try:
        module = __import__(module_name, fromlist=[function_name])
        return getattr(module, function_name)
    except (ImportError, AttributeError):
        return None


def get_module_attribute(module_name, attribute_name):
    """Gets a module attribute by name.

    Args:
        module_name (str): The name of the module.
        attribute_name (str): The name of the attribute.

    Returns:
        The attribute value, or None if not found.
    """
    try:
        module = __import__(module_name)
        return getattr(module, attribute_name)
    except (ImportError, AttributeError):
        return None


def list_class_methods(cls):
    """Lists the methods of a class.

    Args:
        cls (type): The class to inspect.

    Returns:
        A list of method names (strings).
    """
    return [name for name, member in inspect.getmembers(cls) if inspect.isfunction(member)]
