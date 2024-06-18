from setuptools import setup, find_packages
import os

# Assuming static versioning for simplification, but you can reintegrate dynamic versioning if needed
version = '1.0.0'

def readme_contents():
    readme_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')
    with open(readme_path, 'r', encoding='utf-8') as readme_file:  # Added 'encoding' parameter for Python 3
        return readme_file.read()

setup(
    name='trcrpm',
    version=version,  # Use the static version defined above
    description='Temporally-Reweighted CRP Mixture: A nonparametric Bayesian method for multivariate time series',
    long_description=readme_contents(),
    long_description_content_type='text/markdown',  # Specify the content type of the long description
    url='https://github.com/l1aran/trcrpm_sub',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',  # Updated to indicate Python 3 compatibility
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    packages=find_packages(where='src'),  # Simplified package finding
    package_dir={'': 'src'},  # Specify that packages are under 'src' directory
    zip_safe=False,
)

