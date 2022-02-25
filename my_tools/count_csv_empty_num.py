import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='xxxx')
    parser.add_argument('--csv', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.csv, "r") as r:
        lines = r.readlines()
        empty_num = 0
        for line in lines:
            _id, content = line.strip("\n").split(",")
            if len(content) < 3:
                empty_num += 1
        print("empty num is", empty_num)


if __name__ == '__main__':
    main()
