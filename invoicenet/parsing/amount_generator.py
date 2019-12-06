import os
import random
from tqdm import tqdm


def main():
    train_numbers = random.sample(range(0, 10000), 5000)
    train_numbers += random.sample(range(0, 100000), 5000)
    train_numbers += random.sample(range(0, 1000000), 5000)
    for idx in range(1, 16):
        train_numbers += random.sample(range(0, 1000), 1000)

    test_numbers = random.sample(range(0, 1000), 1000)
    test_numbers += random.sample(range(0, 1000), 1000)
    test_numbers += random.sample(range(0, 10000), 1000)
    test_numbers += random.sample(range(0, 100000), 500)
    test_numbers += random.sample(range(0, 1000000), 500)

    random.shuffle(train_numbers)
    random.shuffle(test_numbers)

    os.makedirs(os.path.join("data", "amount"), exist_ok=True)

    for phase, numbers in [("train", train_numbers), ("valid", test_numbers)]:

        with open(os.path.join("data", "amount", "{}.tsv".format(phase)), "w") as fp:

            for number in tqdm(numbers, total=len(numbers)):

                if random.sample(range(1, 10000), 1)[0] % 20 == 0:
                    if random.sample(range(1, 10000), 1)[0] % 10 == 0:
                        fp.write(str(number) + "\t" + str(number) + ".00" + "\n")
                    else:
                        exp = "{}".format(random.sample(range(0, 100), 1)[0])
                        if len(exp) == 1:
                            exp = "0" + exp
                        fp.write(str(number) + "." + exp + "\t" + str(number) + "." + exp + "\n")

                else:
                    fnum = str(number)
                    if len(fnum) > 3:
                        fnum = fnum[:-3] + "," + fnum[-3:]
                        if len(fnum) == 7:
                            if random.sample(range(1, 10000), 1)[0] % 2 == 0:
                                fnum = fnum[0] + ',' + fnum[1:]

                    if random.sample(range(1, 10000), 1)[0] % 20 == 0:
                        fp.write(fnum + "\t" + str(number) + ".00" + "\n")
                    else:
                        exp = "{}".format(random.sample(range(0, 100), 1)[0])
                        if len(exp) == 1:
                            exp = "0" + exp
                        fp.write(fnum + "." + exp + "\t" + str(number) + "." + exp + "\n")


if __name__ == '__main__':
    main()
