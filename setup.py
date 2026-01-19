"""Setup configuration for H2Q-Evo package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
README_PATH = Path(__file__).parent / "README.md"
LONG_DESCRIPTION = README_PATH.read_text(encoding="utf-8")

# Read requirements
REQUIREMENTS_PATH = Path(__file__).parent / "requirements.txt"
REQUIREMENTS = REQUIREMENTS_PATH.read_text(encoding="utf-8").strip().split("\n")

setup(
    name="h2q-evo",
    version="0.1.0",
    description="Quaternion-Fractal Self-Improving Framework for AGI",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="H2Q-Evo Contributors",
    author_email="your-email@example.com",
    url="https://github.com/yourusername/H2Q-Evo",
    license="MIT",
    packages=find_packages(where="h2q_project"),
    package_dir={"": "h2q_project"},
    python_requires=">=3.8",
    install_requires=REQUIREMENTS,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.930",
            "pre-commit>=2.15",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "gpu": [
            "torch-cuda>=11.8",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords=[
        "AGI",
        "quaternion",
        "fractal",
        "holomorphic",
        "online-learning",
        "edge-computing",
        "neural-network",
        "self-improving",
    ],
    project_urls={
        "Documentation": "https://github.com/yourusername/H2Q-Evo/blob/main/README.md",
        "Source": "https://github.com/yourusername/H2Q-Evo",
        "Issues": "https://github.com/yourusername/H2Q-Evo/issues",
        "Discussions": "https://github.com/yourusername/H2Q-Evo/discussions",
    },
    include_package_data=True,
    zip_safe=False,
)
