from setuptools import setup

setup(
    name="ue5osc",
    version="1.0",
    description="This is a wrapper for communicating commands to Unreal Engine 5+ using OSC.",
    author="Anthony J. Clark, Anjali Nuggehalli, and Francisco Morales Puente",
    license="MIT",
    packages=["ue5osc"],
    install_requires=["python-osc==1.8.3"],
    # TODO: does zip_safe do anything?
    zip_safe=False,
)
