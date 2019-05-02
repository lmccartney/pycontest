import numpy as np
import os

from hypothesis import given, strategies as st

from pycontest import simulation as sim2d
from pycontest.utils import momentum, E_kin


@given(mass1=st.floats(min_value=.1, max_value=1e3),
       mass2=st.floats(min_value=.1, max_value=1e3))
def test_energy_hypothesis(mass1, mass2):
    domain = ([-2, 12], [0, 3])
    dt = 0.5
    t_max = 6
    loc_0 = np.array([[0, 1.5], [10, 1.5]])
    vel_0 = np.array([[1, 0], [-1, 0]])
    radius = 1
    mass = [mass1, mass2]

    loc, vel = sim2d.simulation(t_max, dt, mass, radius, loc_0, vel_0, domain)

    # save reference simulations (only once when the test is created)
    np.save(
        os.path.join(os.path.dirname(__file__), "../data/locations_ref.npy"),
        loc)

    # read in reference values (every time the test is run)
    loc_ref = np.load(
        os.path.join(os.path.dirname(__file__), "../data/locations_ref.npy"))

    # from scipy docs:
    # The tolerance values are positive, typically very small numbers.
    # The relative difference (rtol * abs(b)) and the absolute difference atol
    # are added together to compare against the absolute difference
    # between a and b.
    np.testing.assert_allclose(loc, loc_ref, atol=0, rtol=1e-9,
                               err_msg="for locations")
