# InvoiceNet
Deep neural network to extract intelligent information from PDF invoice documents.


## Training

Prepare the data for training first by running the following command:
```
python3 prepare_data.py --data_dir data/
```

Train InvoiceNet using the following command:
```
PYTHONPATH="$PWD" python3 invoicenet/acp/train.py [field]
```


## Prediction

Prepare the data for prediction first by running the following command:
```
python3 prepare_data.py --data_dir data/ --prediction
```

Run InvoiceNet using the following command:
```
PYTHONPATH="$PWD" python3 invoicenet/acp/predict.py [field]
```
