import setuptools

setuptools.setup(
    name="AutoSMOTE",
    version='1.0.0',
    author="Daochen Zha",
    author_email="daochen.zha@rice.edu",
    description="Code for AutoSMOTE",
    url="https://github.com/daochenzha/autosmote",
    keywords=["Reinforcement Learning"],
    packages=setuptools.find_packages(exclude=('tests',)),
    requires_python='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
