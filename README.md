# InvoiceNet
Classification of entities extracted from invoice scans

## Instructions
Two models have been implemented for the purpose of invoice data classification.

The first one is a convolution neural network which trains on custom word embeddings trained using word2vec.
```
python3 main.py -h
```

The second model is based on CloudScan.
```
python3 cloud_scan.py -h
```

#### Dependencies:
```
python3

nltk
pandas
numpy
keras
pickle
gzip
gensim
sklearn
matplotlib
tqdm
datefinder
```

## References: 

- Rasmus Berg Palm, Ole Winther, Florian Laws, CloudScan - A configuration-free invoice analysis system using recurrent neural networks,  	[arXiv:1708.07403v1 [cs.CL]](https://arxiv.org/pdf/1708.07403.pdf)
