from setuptools import setup

setup(
    name="boxnav",
    version="1.0",
    description="Creates a box-based simulation environment for navigation tasks.",
    author="Anthony J. Clark, Anjali Nuggehalli, Francisco Morales Puente, Kellie Au, and Tommy Ryan",
    license="MIT",
    packages=["boxnav"],
    install_requires=["ue5osc", "matplotlib"],
)
