import numpy as np
import pytest

from pycontest.transport import transport


def test_simple_transport():
    """Simple test for our transport method"""
    loc = 3
    vel = 1
    dt = .5
    assert transport(loc, vel, dt) == 3.5


def test_simple_array():
    """Test our method works with a simple array"""
    loc = np.array([[1, 2], [11, 12]])
    vel = np.array([[1, 1], [-1, -1]])
    dt = 1
    assert (transport(loc, vel, dt) == np.array([[2, 3], [10, 11]])).all()


# let's try one more, with dt that is float
def test_transport_3():
    loc = np.array([[1, 2], [11, 12]])
    vel = np.array([[1, 1], [-1, -1]])
    dt = .5
    assert (transport(loc, vel, dt) == np.array(
        [[1.5, 2.5], [10.5, 11.5]])).all()


# # and now let's try with lists
def test_transport_4():
    loc = [[1, 2], [11, 12]]
    vel = [[1, 1], [-1, -1]]
    dt = .5
    assert (transport(loc, vel, dt) == np.array(
        [[1.5, 2.5], [10.5, 11.5]])).all()
