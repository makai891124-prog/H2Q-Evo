from h2q_project.utils.reflection import get_class_from_string, list_class_methods
import inspect

class SelfReflectionModule:
    """A lightweight self-reflection module for H2Q components."""

    def __init__(self, target_module_name, target_class_name=None):
        """Initializes the SelfReflectionModule.

        Args:
            target_module_name (str): The name of the module to reflect on (e.g., 'h2q_project.core.h2q_kernel').
            target_class_name (str, optional): The name of the class to reflect on within the module. Defaults to None.
                                             If None, the module itself is inspected.
        """
        self.target_module_name = target_module_name
        self.target_class_name = target_class_name
        self.target = None # The reflected object, module or class

        self._load_target()

    def _load_target(self):
        """Loads the target module or class using reflection."""
        if self.target_class_name:
            self.target = get_class_from_string(self.target_module_name, self.target_class_name)
            if self.target is None:
                raise ValueError(f"Class '{self.target_class_name}' not found in module '{self.target_module_name}'")
        else:
            try:
                self.target = __import__(self.target_module_name)
            except ImportError:
                raise ValueError(f"Module '{self.target_module_name}' not found.")


    def describe(self):
        """Returns a description of the target module or class.

        Returns:
            str: A string describing the target.
        """
        if self.target_class_name:
            return self._describe_class()
        else:
            return self._describe_module()

    def _describe_module(self):
        """Describes a module.

        Returns:
            str: A description of the module.
        """
        return f"Module: {self.target_module_name}"

    def _describe_class(self):
        """Describes a class.

        Returns:
            str: A description of the class, including its methods and docstring.
        """
        description = f"Class: {self.target_class_name} in Module: {self.target_module_name}\n"
        description += f"Docstring: {inspect.getdoc(self.target)}\n"
        methods = list_class_methods(self.target)
        description += f"Methods: {methods}"
        return description

    def list_methods(self):
        """Lists the methods of the target class (if applicable).

        Returns:
            list: A list of method names (strings), or None if the target is not a class.
        """
        if self.target_class_name:
            return list_class_methods(self.target)
        else:
            return None
