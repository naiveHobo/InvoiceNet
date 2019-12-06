# InvoiceNet
Deep neural network to extract intelligent information from PDF invoice documents.


## Dependencies

Before running InvoiceNet, you need to install a few dependencies:
- [tesseract-ocr](https://github.com/tesseract-ocr/tesseract/wiki)
- [poppler](https://poppler.freedesktop.org/)


## Using the GUI

InvoiceNet provides you with a GUI to train a model on your data and extract information from invoice documents using this trained model

Run the following command to run the trainer GUI:

```
python3 trainer.py
```

Run the following command to run the extractor GUI:

```
python3 extractor.py
```

You need to prepare the data for training and extraction first. 
You can do so by setting the **Data Folder** field to the directory containing your training/prediction data and the clicking the **Prepare Data** button.
Once the data is prepared, you can start training/prediction by clicking the **Start** button.


## Using the CLI

### Training 

Your training data should be in the following format:

```
train_data/
    invoice1.pdf
    invoice1.json
    invoice2.pdf
    invoice2.json
    ...
```

The JSON labels should have the following format:
```
{
 "amount": "6,245.80",
 "number": "7402488304"
 ... other fields
}
```

Prepare the data for training first by running the following command:
```
python3 prepare_data.py --data_dir train_data/
```

Train InvoiceNet using the following command:
```
python3 train.py --field [amount|date|number] --batch_size 8
```

---

### Prediction

For extracting information using the trained InvoiceNet model, you just need to place the PDF invoice documents in one directory in the following format:

```
predict_data/
    invoice1.pdf
    invoice2.pdf
    ...
```

Prepare the data for prediction first by running the following command:
```
python3 prepare_data.py --data_dir predict_data/ --prediction 
```

Run InvoiceNet using the following command:
```
python3 predict.py --field [amount|date|number]
```
