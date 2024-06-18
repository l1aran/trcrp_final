from setuptools import setup, find_packages
import os

def get_version():
    version = "1.0.0"
    return version

pkg_version = get_version()

setup(
    name='cgpm',
    version=pkg_version,
    description='GPM Crosscat',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    url='https://github.com/l1aran/cgpm_sub',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    packages=[
        'cgpm',
        'cgpm.crosscat',
        'cgpm.dummy',
        'cgpm.factor',
        'cgpm.kde',
        'cgpm.knn',
        'cgpm.mixtures',
        'cgpm.network',
        'cgpm.primitives',
        'cgpm.regressions',
        'cgpm.tests',
        'cgpm.uncorrelated',
        'cgpm.utils',
        'cgpm.venturescript',
    ],
    package_dir={
        'cgpm': 'src',
        'cgpm.tests': 'tests',
    },
    package_data={
        'cgpm.tests': ['graphical/resources/satellites.csv'],
    },
    install_requires=[
        'numpy',
        'pandas',
        'cython',
    ],
    tests_require=[
        'pytest',
    ],
)
