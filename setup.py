import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='InvoiceNet',
                 version='0.1',
                 description='A deep neural network to extract intelligent information from invoice documents',
                 url='https://github.com/naiveHobo/InvoiceNet',
                 author='naiveHobo',
                 author_email='sarthakmittal2608@gmail.com',
                 license='MIT',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 packages=setuptools.find_packages(),
                 install_requires=[
                    "datefinder==0.7.1",
                    "future==0.18.2",
                    "numpy==1.19.1",
                    "opencv-python==4.3.0.36",
                    "pdf2image==1.13.1",
                    "Pillow==7.2.0",
                    "pytesseract==0.3.4",
                    "python-dateutil==2.8.1",
                    "PyYAML==5.3.1",
                    "scipy==1.5.2",
                    "simplejson==3.17.2",
                    "six==1.15.0",
                    "torch==1.6.0",
                    "tqdm==4.48.0"
                 ])
