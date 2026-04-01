"""
安装脚本：用于开发和测试
"""

from setuptools import setup, find_packages

setup(
    name="turboquantx",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "full": [
            "torch>=2.0.0",
            "triton>=3.0.0", 
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "bitsandbytes>=0.41.0",
            "matplotlib>=3.7.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ]
    },
    python_requires=">=3.10",
)