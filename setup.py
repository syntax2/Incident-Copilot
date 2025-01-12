# setup.py
from setuptools import setup, find_packages

setup(
    name="incident-copilot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "pandas==2.1.3",
        "numpy==1.26.2",
        "scikit-learn==1.3.2",
        "transformers==4.35.2",
        "torch==2.1.1",
        "python-dotenv==1.0.0",
        "pytest==7.4.3",
        "black==23.11.0",
        "flake8==6.1.0"
    ],
    python_requires=">=3.8",
)