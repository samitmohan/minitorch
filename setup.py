from setuptools import setup, find_packages

setup(
    name='minitorch',
    version='0.1.0',
    description='A miniature PyTorch-like autograd engine built from scratch',
    author='Samit Mohan',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib'
    ],
)
