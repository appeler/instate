"""Concatenates gz part files
Usage: python concater.py -o concatenate.tar.gz -f file_1.tar.gz.partaa file_2.tar.gz.partab file_3.tar.gz.partac
"""
import sys
import argparse
import os

base_dir = "/data/in-rolls/parsed/"


def concatenate(args):
    with open(os.path.join(base_dir, args.out), "wb") as f:
        for fname in args.files:
            with open(os.path.join(base_dir, fname), "rb") as g:
                f.write(g.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Join gz files together",
    )
    parser.add_argument("-f", "--files", nargs="+", default=[])
    parser.add_argument("-o", "--out")
    args = parser.parse_args()
    concatenate(args)
