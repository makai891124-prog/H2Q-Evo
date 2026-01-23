import unittest
from h2q_project.code_style_checker import CodeStyleChecker

class ValidationPipeline:
    def __init__(self, steps=None):
        self.steps = steps if steps is not None else []

    def add_step(self, step):
        self.steps.append(step)

    def run(self):
        for step in self.steps:
            if not step.run():
                return False
        return True

class CodeStyleCheckStep:
    def __init__(self, target_directory='h2q_project', linter='flake8'):
        self.checker = CodeStyleChecker(target_directory, linter)

    def run(self):
        return self.checker.check_all_files()


if __name__ == '__main__':
    pipeline = ValidationPipeline()
    pipeline.add_step(CodeStyleCheckStep())

    if pipeline.run():
        print("Validation pipeline passed!")
    else:
        print("Validation pipeline failed.")
