from setuptools import setup, find_packages

setup(
    name='h2q_project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List any dependencies here.  For example:
        # 'requests >= 2.20',
    ],
    entry_points={
        'console_scripts': [
            # Define any command-line scripts here. For example:
            # 'my_script = my_package.my_module:main',
        ],
    },
)