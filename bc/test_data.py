import pdb

import numpy as np

from battlecruiser.data import MemoryDataset


def test_consistent():
    data1 = [1, 2, 3]
    data2 = np.random.rand(3, 5)
    data3 = (4, 5, 6)
    ds = MemoryDataset(data1, data2, data3)
    pdb.set_trace()


def test_consistent2():
    data = np.random.rand(3, 5)
    ds = MemoryDataset(data)
    pdb.set_trace()


def test_inconsistent():
    data1 = [1, 2, 3, 4]
    data2 = np.random.rand(3, 5)
    ds = MemoryDataset(data1, data2)


if __name__ == "__main__":
    # test_consistent()
    test_consistent2()
    # test_inconsistent()
