import argparse
import glob
import os
import pdb

import spacy
from spacy.matcher import Matcher


def redact_file(file, args):
    with open(file, 'r') as f:
        text = f.read()
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = [token.text for token in doc]

    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
    # pdb.set_trace()

    pattern = [{
        "ORTH": "("
    }, {
        "SHAPE": "ddd"
    }, {
        "ORTH": ")"
    }, {
        "SHAPE": "ddd"
    }, {
        "ORTH": "-",
        "OP": "?"
    }, {
        "SHAPE": "dddd"
    }]

    matcher = Matcher(nlp.vocab)
    matcher.add("PhoneNumber", [pattern])
    matches = matcher(doc)
    pdb.set_trace()


def redact_all_files(args):
    file_lists = []
    for pattern in args.input:
        file_lists.extend(glob.glob(pattern))
    for file in file_lists:
        redact_file(file, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        type=str,
                        required=True,
                        help="Input files",
                        action="append")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="destination")
    parser.add_argument("--names", action="store_true", help="redact names.")
    parser.add_argument("--dates", action="store_true", help="redact dates.")
    parser.add_argument("--phones", action="store_true", help="redact phones.")
    parser.add_argument("--gender", action="store_true", help="redact gender.")
    parser.add_argument("--concept",
                        type=str,
                        help="redact related concepts",
                        action='append')
    parser.add_argument("--stats", type=str, help="stats output.")
    args = parser.parse_args()
    redact_all_files(args)
