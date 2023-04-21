import os
import setuptools

this_directory = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as f:
    long_description = f.read()

# read the contents of requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name = 'PyTrial',
    version = '0.0.4',
    author = 'Zifeng Wang, Brandon Theodoru, Tianfan Fu, Jingtang Ma, Jimeng Sun',
    author_email = 'zifengw2@illinois.edu',
    description = 'PyTrial: A Python Package for Artificial Intelligence in Drug Development',
    url = 'https://github.com/RyanWangZf/pytrial',
    keywords=['drug development', 'clinical trial', 'artificial intelligence', 'deep learning', 'machine learning'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)