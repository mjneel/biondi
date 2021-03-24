import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='biondi',
    version='0.1.4',
    author='Michael Neel',
    author_email='neelm@uci.edu',
    description='Machine learning code and resources for analyzing choroid plexus pathology',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mjneel/Biondi',
    license='GNU GPLv3',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)
