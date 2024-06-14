from setuptools import setup, find_packages

setup(
    name="evil",
    version="0.1.0",
    description="Implementation of the EvIL algorithm for IRL.",
    author="Silvia Sapora",
    author_email="silvia.sapora@gmail.com",
    packages=find_packages(exclude=("tests")),
)

print(find_packages(exclude=("tests")))
