from setuptools import setup, find_packages

setup(
    name='PiTensor',
    version='0.1.0',
    author='Marco Lo Pinto',
    author_email='marcolopinto.dev@gmail.com',
    packages=find_packages(),
    url='https://github.com/MarcoLoPinto/PiTensor',
    license='AGPL-3.0',
    description='A simple and educational deep learning framework from scratch.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "tqdm",
    ],
    python_requires='>=3.9'
)
