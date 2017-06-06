import re
import string
import numpy as np

NULL_CHAR = '\0'


VALID_CHARS = [x for x in string.printable+'\0']

WHITESPACE_CHARS = [x for x in string.whitespace if x in VALID_CHARS]

NON_VALID_REGEX = re.compile('[^' + ''.join(VALID_CHARS) + ']')

INT2CHAR_MAP = dict(zip(np.arange(len(VALID_CHARS)), VALID_CHARS))

CHAR2INT_MAP = {val: key for key, val in INT2CHAR_MAP.items()}

def ints2str(integers):
    return ''.join([INT2CHAR_MAP[x] for x in integers])

def str2ints(s):
    return [CHAR2INT_MAP[c] for c in s]

WHITESPACE_INTS = str2ints(''.join(WHITESPACE_CHARS))