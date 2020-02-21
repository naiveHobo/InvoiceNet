#!/usr/bin/env bash
# install python3 and pip3
sudo yum install -y https://centos7.iuscommunity.org/ius-release.rpm
sudo yum update
sudo yum install -y python36u python36u-libs python36u-devel python36u-pip
python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'

# install tesseract-ocr
sudo yum install yum-utils
sudo yum-config-manager --add-repo https://download.opensuse.org/repositories/home:/Alexander_Pozdnyakov/CentOS_7/
sudo rpm --import https://build.opensuse.org/projects/home:Alexander_Pozdnyakov/public_key
sudo yum update
sudo yum install tesseract

# install dependencies
sudo yum install poppler-utils libXext libSM libXrender

# install virtualenv
pip3 install --user virtualenv

virtualenv env -p python3
source env/bin/activate

pip install .