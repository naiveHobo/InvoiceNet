import argparse
from invoicenet.acp.acp import AttendCopyParse


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--field", type=str, required=True, choices=["vendorname", "invoicedate", "invoicenumber", "amountnet", "amounttax", "amounttotal", "vatrate", "vatid", "taxid", "iban", "bic"],
                    help="field to train parser for")
    ap.add_argument("--data_dir", type=str, default='processed_data/',
                    help="path to directory containing prepared data")
    ap.add_argument("--pred_dir", type=str, default='predictions/',
                    help="path to directory containing prepared data")

    args = ap.parse_args()

    print("Training...")
    model = AttendCopyParse(field=args.field, data_dir=args.data_dir, batch_size=1, restore=True)
    model.test_set(out_path=args.pred_dir)


if __name__ == '__main__':
    main()
