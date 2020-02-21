import os
import random
import argparse
import datetime
from tqdm import tqdm

years = list(range(2000, 2100))
months = list(range(1, 13))
days = list(range(1, 32))

year_formats = ['%y', '%Y']
month_formats = ['%m', '%b', '%B']
day_formats = ['%d']

date_formats = [
    'day/month/year',
    'month/day/year',
    'year/month/day',
    'month/year/day',
    'year/day/month',
    'day/year/month',
    'day-month-year',
    'month-day-year',
    'year-month-day',
    'month-year-day',
    'year-day-month',
    'day-year-month',
    'day.month.year',
    'month.day.year',
    'year.month.day',
    'month.year.day',
    'year.day.month',
    'day.year.month',
    'day\\month\\year',
    'month\\day\\year',
    'year\\month\\day',
    'month\\year\\day',
    'year\\day\\month',
    'day\\year\\month',
    'day month, year',
    'month day, year',
    'year, month day',
    'year, day month',
    'year day month',
    'day month year',
    'month day year'
]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_size", type=int, default=40000,
                    help="number of training samples to generate")
    ap.add_argument("--test_size", type=int, default=10000,
                    help="number of test samples to generate")

    args = ap.parse_args()

    train_days = random.choices(days, k=args.train_size)
    train_months = random.choices(months, k=args.train_size)
    train_years = random.choices(years, k=args.train_size)
    train_day_formats = random.choices(day_formats, k=args.train_size)
    train_month_formats = random.choices(month_formats, k=args.train_size)
    train_year_formats = random.choices(year_formats, k=args.train_size)
    train_date_formats = random.choices(date_formats, k=args.train_size)

    test_days = random.choices(days, k=args.test_size)
    test_months = random.choices(months, k=args.test_size)
    test_years = random.choices(years, k=args.test_size)
    test_day_formats = random.choices(day_formats, k=args.test_size)
    test_month_formats = random.choices(month_formats, k=args.test_size)
    test_year_formats = random.choices(year_formats, k=args.test_size)
    test_date_formats = random.choices(date_formats, k=args.test_size)

    os.makedirs(os.path.join("data", "date"), exist_ok=True)

    with open(os.path.join("data", "date", "train.tsv"), "w") as fp:
        for sample in tqdm(zip(train_date_formats,
                               train_year_formats, train_month_formats, train_day_formats,
                               train_years, train_months, train_days), total=args.train_size):
            if sample[5] not in [1, 3, 5, 7, 8, 10, 12]:
                if sample[5] == 2:
                    day = random.sample(range(1, 29), k=1)[0]
                else:
                    day = random.sample(range(1, 31), k=1)[0]
            else:
                day = sample[6]
            date = datetime.date(sample[4], sample[5], day)
            fp.write(date.strftime(
                sample[0].replace('year', sample[1]).replace('month', sample[2]).replace('day', sample[3])))
            fp.write('\t')
            fp.write(date.strftime('%m-%d-%Y'))
            fp.write('\n')

    with open(os.path.join("data", "date", "valid.tsv"), "w") as fp:
        for sample in tqdm(zip(test_date_formats,
                               test_year_formats, test_month_formats, test_day_formats,
                               test_years, test_months, test_days), total=args.test_size):
            if sample[5] not in [1, 3, 5, 7, 8, 10, 12]:
                if sample[5] == 2:
                    day = random.sample(range(1, 29), k=1)[0]
                else:
                    day = random.sample(range(1, 31), k=1)[0]
            else:
                day = sample[6]
            date = datetime.date(sample[4], sample[5], day)
            fp.write(date.strftime(
                sample[0].replace('year', sample[1]).replace('month', sample[2]).replace('day', sample[3])))
            fp.write('\t')
            fp.write(date.strftime('%m-%d-%Y'))
            fp.write('\n')


if __name__ == '__main__':
    main()
