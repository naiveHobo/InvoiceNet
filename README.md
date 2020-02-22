# InvoiceNet
Deep neural network to extract intelligent information from PDF invoice documents.

## Installation
To install InvoiceNet, run the following commands (CentOS 7):
```
git clone https://github.com/naiveHobo/invoicenet-gbr.git
cd invoicenet-gbr/

# Run installation script
./install.sh

# Source virtual environment
source env/bin/activate
```
The install.sh script will install all the dependencies, create a virtual enviroment, and install InvoiceNet in the virtual environment.

## Data Preparation
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
 "vendorname":"Hetzner Online GmbH",
 "invoicedate":"12-01-2017",
 "invoicenumber":"R0007546449",
 "amounttotal":"137.51",
 ... other fields
}
```

To begin the data preparation process, click on the "Prepare Data" button in the GUI or follow the instructions below if you're using the CLI.


## Using the GUI

InvoiceNet provides you with a GUI to train a model on your data and extract information from invoice documents using this trained model

Run the following command to run the trainer GUI:

```
python trainer.py
```

Run the following command to run the extractor GUI:

```
python extractor.py
```

You need to prepare the data for training and extraction first. 
You can do so by setting the **Data Folder** field to the directory containing your training/prediction data and the clicking the **Prepare Data** button.
Once the data is prepared, you can start training/prediction by clicking the **Start** button.


## Using the CLI

### Training 

Prepare the data for training first by running the following command:
```
python prepare_data.py --data_dir train_data/
```

Train InvoiceNet using the following command:
```
python train.py --field [vendorname|invoicedate|invoicenumber|amountnet|amounttax|amounttotal|vatrate|vatid|taxid|iban|bic] --batch_size 8
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
python prepare_data.py --data_dir predict_data/ --prediction 
```

Run InvoiceNet using the following command:
```
python predict.py --field [vendorname|invoicedate|invoicenumber|amountnet|amounttax|amounttotal|vatrate|vatid|taxid|iban|bic]
```
