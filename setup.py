from setuptools import find_packages, setup

with open("requirements-cpu.txt", "r") as f:
    requirements = list(f.read().splitlines())

setup(
    name="reqap",
    version="1.0",
    description="Code for the ReQAP project.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Philipp Christmann",
    author_email="pchristm@mpi-inf.mpg.de",
    url="https://reqap.mpi-inf.mpg.de",
    packages=find_packages(),
    include_package_data=False,
    keywords=[
        "qa",
        "question answering",
        "heterogeneous QA"
    ],
    classifiers=["Programming Language :: Python :: 3.12"],
    install_requires=requirements,
)