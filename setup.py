"""
Setup script for scReGAT package
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
            return requirements
    return []

setup(
    name='scregat',
    version='0.1.0',
    description='Single-cell Regulatory Graph Attention Network for ATAC-seq data analysis',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='scReGAT Team',
    author_email='',
    url='https://github.com/yourusername/scReGAT',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='single-cell ATAC-seq graph neural network gene expression prediction',
    include_package_data=True,
    package_data={
        'scregat': [],
    },
    entry_points={
        'console_scripts': [
            # Add command-line scripts here if needed
        ],
    },
)

