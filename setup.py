import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='invoicenet',
    version='0.1',
    author="Sarthak Mittal",
    author_email="sarthakmittal2608@gmail.com",
    description="A deep neural network to extract information from invoice documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=["tqdm",
                      "pdf2image",
                      "tensorflow==2.0.0rc1",
                      "pytesseract",
                      "numpy",
                      "sklearn",
                      "chars2vec",
                      "gensim",
                      "pyyaml",
                      "opencv-python",
                      "keras"]
)
