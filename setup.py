import pathlib
from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="taug",
    version="0.1.0",
    description="Time Series Forecasting and Data Augmentation using Deep Generative Models ",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/amirabbasasadi/taug",
    author="Amirabbas Asadi",
    author_email="amir137825@gmail.com",
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=["tensorflow>=2.0.0", "keras", "tensorflow_addons", "sktime>=0.7.0"],
)
