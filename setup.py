import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='InvoiceNet',
                 version='0.1',
                 description='A deep neural network to extract intelligent information from invoice documents',
                 url='https://github.com/naiveHobo/invoiceNet--parserr',
                 author='naiveHobo',
                 author_email='sarthakmittal2608@gmail.com',
                 license='MIT',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 packages=setuptools.find_packages(),
                 install_requires=[
                     "tensorflow==1.14",
                     "tqdm",
                     "pdf2image",
                     "pytesseract",
                     "pyyaml",
                     "opencv-python",
                     "keras",
                     "simplejson"
                 ])
