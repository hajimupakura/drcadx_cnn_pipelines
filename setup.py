import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="drcadx_cnn",
    version="0.1",
    author="Haji Mupakura",
    author_email="haji.mupakura@drcadx.com",
    description="Top Recipes",
    install_requires=[
        "py4j==0.10.4",
        "pyspark==3.1.3",
            ],
    packages=find_packages(),
)