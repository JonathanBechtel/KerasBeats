import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name= "keras-beats",
    version= "0.0.1",
    author= "Jonathan Bechtel",
    author_email= "jonathan@jonathanbech.tel",
    description= "Lightweight installation of NBeats NN architecture for keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JonathanBechtel/KerasBeats",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages = ['kerasbeats'],
    python_requires=">=3.6",
    install_requires = ['tensorflow>=2.0.0', 'pandas', 'numpy']
)