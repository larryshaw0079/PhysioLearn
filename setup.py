import os

from setuptools import setup, find_packages


def __parse_requirements(file_name):
    with open(file_name, 'r') as f:
        line_striped = (line.strip() for line in f.readlines())
    requirements = [line for line in line_striped if line and not line.startswith('#')]
    # requirements.append(__check_pytorch())
    return requirements


# Get description from README
root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="physiolearn",
    version="0.0.1a2",
    author="qinfeng xiao",
    author_email="qfxiao@bjtu.edu.cn",
    description="PhysioLearn: A Python Toolbox for Physiological Time-series Deep Learning",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/larryshaw0079/PhysioSSL",
    install_requirements=__parse_requirements('requirements.txt'),
    install_requires=__parse_requirements('requirements.txt'),
    packages=find_packages(),
    python_requires=">=3.6",
    license='MIT license'
)
