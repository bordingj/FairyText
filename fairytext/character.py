import re
import string
import numpy as np
import pandas as pd

NULL_CHAR = '\0'


VALID_CHARS = [x for x in string.printable+'\0']

WHITESPACE_CHARS = [x for x in string.whitespace if x in VALID_CHARS]

NON_VALID_REGEX = re.compile('[^' + ''.join(VALID_CHARS) + ']')

CHAR2INT_MAP = pd.Series(np.arange(len(VALID_CHARS)).astype(np.int16), index=VALID_CHARS)

INT2CHAR_MAP = pd.Series(CHAR2INT_MAP.index.values, index=CHAR2INT_MAP.values)

def ints2str(integers):
    return ''.join(INT2CHAR_MAP.loc[INT2CHAR_MAP].values)

def str2ints(str_):
    return CHAR2INT_MAP.loc[list(str_)].values

WHITESPACE_INTS = str2ints(''.join(WHITESPACE_CHARS))