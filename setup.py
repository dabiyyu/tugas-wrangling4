import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name = "car-prediction",
    version = "0.0.1",
    author = "author",
    autor_email = "author@gmail.com",
    description = "simple repo for exercise",
    long_description = "long_description",
    long_description_content_type = "text/markdown",
    url = "",
    packages = setuptools.find_packages(),
    classifier = ["Programming Language :: Python :: 3"],
    install_requires = [
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "seaborn"],
    python_requires = ">=3.7"
)