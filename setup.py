from setuptools import setup, find_packages

setup(
    name='kmeanswithnulls',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    author='Arthur Sedek',
    author_email='arthur.sedek@gmail.com',
    description=' a Python implementation of the KMeans clustering algorithm which includes support for handling missing values in the dataset',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/aasedek/KmeansWithNulls.git',
)
