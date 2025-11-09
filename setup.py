from setuptools import setup, find_packages

setup(
    name="InferenceModel",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "accelerate",
        "triton"
    ],
)