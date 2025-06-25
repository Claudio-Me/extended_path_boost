from setuptools import setup, find_packages

setup(
    name='Extended_Path_Boost',
    version='1.2',
    packages=['tests', 'tests.datasets_used_for_tests', 'tests.test_extended_path_boost',
              'tests.test_sequential_path_boost', 'tests.tests_extended_boosting_matrix', 'extended_path_boost',
              'extended_path_boost.utils', 'extended_path_boost.utils.classes',
              'extended_path_boost.utils.classes.interfaces'],
    url='',
    license='MIT License',
    author='Claudio',
    classifiers=[
        "license ::This template covers installation, usage, features, and links to documentation and contribution guidelines. Adjust as needed for your project specifics.This template covers installation, usage, features, and links to documentation and contribution guidelines. Adjust as needed for your project specifics. MIT License",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Gradient Boosting",
    ],
    author_email='claudiomeggio@gmail.com',
    description='gradient boost on graph data'
)
