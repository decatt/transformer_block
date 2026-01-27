from setuptools import setup, find_packages

setup(
    name="transformer_block",
    version="0.1.0",
    description="Transformer block with KV cache and Mixture of Experts support",
    author="decatt",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
)
