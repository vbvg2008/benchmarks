import pdb
import sys

import numpy as np


class A:
    def __init__(self, name):
        self.name = name

    def reset_name(self):
        del self.name
        self.name = "fsaf"


a = A(name="a")
print(a.name)

a.reset_name()
print(a.name)
