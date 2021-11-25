import pathlib
from setuptools import find_packages, setup

# The directory containing the file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="AugmentTS",
    version="0.1.0",
    description="Time Series Forecasting and Data Augmentation using Deep Generative Models ",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/DrSasanBarak/AugmentTS",
    author="Sasan Barak",
    author_email="s.barak@soton.ac.uk",
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=["tensorflow>=2.0.0", "keras", "tensorflow_addons", "sktime>=0.7.0"],
)
