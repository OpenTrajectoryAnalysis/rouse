import os,sys
try: # prevent any plotting
    del os.environ['DISPLAY']
except KeyError:
    pass

import numpy as np
np.seterr(all='raise') # pay attention to details

import unittest
from unittest.mock import patch

from context import rouse

"""
exec "norm jjd}O" | let @a="\n'" | exec "g/^class Test/norm w\"Ayt(:let @a=@a.\"',\\n'\"" | norm i__all__ = ["ap}kcc]kV?__all__j>>
"""
__all__ = [
    'TestModel',
    'TestMSDfun',
]

class myTestCase(unittest.TestCase):
    def assert_array_equal(self, array1, array2):
        try:
            np.testing.assert_array_equal(array1, array2)
            res = True
        except AssertionError as err: # pragma: no cover
            res = False
            print(err)
        self.assertTrue(res)

class TestModel(myTestCase):
    def setUp(self):
        self.model = rouse.Model(5, 1, 1)
        self.model_nosetup = rouse.Model(5, 1, 1, setup_dynamics=False)

    def test_setup(self):
        mod = rouse.Model(5, 1, 1, setup_dynamics=False)
        mod.F[-1] = [1, 0, 0]
        mod.update_dynamics()
        mod.add_tether(0, 1, 0)
        mod.update_dynamics()

        mod.add_tether()
        mod.update_F_only()
        with self.assertRaises(RuntimeError):
            mod.check_dynamics(run_if_necessary=False)
        mod.update_F_only(override_full_update=True)
        mod.check_dynamics(run_if_necessary=False)

        mod = rouse.Model(1)
        self.assert_array_equal(mod.A, np.array([[0.]]))

    def test_operators(self):
        self.assertTrue(self.model == self.model_nosetup)
        self.model_nosetup.F[-1] = 1
        self.assertFalse(self.model == self.model_nosetup)
        self.assertTrue(self.model != self.model_nosetup) # we have != as mixin for == :)
        self.assertFalse(self.model == rouse.Model(6, 1, 1, setup_dynamics=False))

        self.assertEqual(repr(self.model), "rouse.Model(N=5, D=1, k=1, d=3)")
        self.assertEqual(repr(rouse.Model(5, 1, 1, add_bonds=[(2, 4), (1, 3, 0.5)])),
                         "rouse.Model(N=5, D=1, k=1, d=3) with 2 additional bonds")

    def test_check_dynamics(self):
        self.model.check_dynamics(dt=1)

        with self.assertRaises(RuntimeError):
            self.model_nosetup.check_dynamics()

        self.model_nosetup.check_dynamics(dt=1, run_if_necessary=True)
        self.model_nosetup.k = 2
        with self.assertRaises(RuntimeError):
            self.model_nosetup.check_dynamics(dt=1, run_if_necessary=False)

    def test_steady_state(self):
        M, C = self.model_nosetup.steady_state()
        self.assert_array_equal(M, np.zeros_like(M))

    def test_propagate(self):
        _, C0 = self.model.steady_state()
        M0 = np.array(3*[np.linspace(0, 1, self.model.N)]).T
        M1, C1 = self.model.propagate(M0, C0, 0.1)

        self.assertTrue(np.allclose(M1, M0, atol=1))
        self.assertTrue(np.allclose(C1, C0, atol=1))

        self.assertTrue(np.allclose(M1, self.model.propagate_M(M0)))
        self.assertTrue(np.allclose(C1, self.model.propagate_C(C0)))

        self.model.D = 0
        M2, C2 = self.model.propagate(M0, np.zeros_like(C0), 0.1)

        self.assertEqual(self.model._dynamics['D'], self.model.D)
        self.assertTrue(np.allclose(M2, M0, atol=1))
        self.assert_array_equal(C2, np.zeros_like(C2))

    def test_conf_ss(self):
        conf = self.model.conf_ss()
        self.assertTupleEqual(conf.shape, (self.model.N,3))

    def test_evolve(self):
        conf = self.model.conf_ss()
        conf = self.model.evolve(conf)
        self.assertTupleEqual(conf.shape, (5,3))

    def test_contact_frequency(self):
        hic = self.model.contact_frequency()
        self.assertEqual(np.sum(np.isinf(hic)), 5)
        self.assert_array_equal(np.round(np.diagonal(hic, 1), 10), [1, 1, 1, 1])
        self.assert_array_equal(np.round(np.diagonal(hic, 1), 10), np.round(np.diagonal(hic, -1), 10))

    def test_MSD_ACF(self):
        dts = np.array([0, 1, 10, 100, np.inf])
        msd = self.model.MSD(dts, w=[0, 1, 0, -1, 0])
        self.assert_array_equal(np.round(msd[[0, -1]], 10), [0, 2*3*2]) # correct values?
        self.assertAlmostEqual(msd[-1], self.model.MSD(np.inf, w=[0, 1, 0, -1, 0]))

        # The better check for correct values: do MSD and ACF match?
        # (these are computed along separate paths)
        acf = self.model.ACF(dts, w=[0, 1, 0, -1, 0])
        comp = msd - 2*(acf[0] - acf)
        comp[np.abs(comp) < 1e-10] = 0
        self.assert_array_equal(comp, np.zeros_like(comp))
        self.assertAlmostEqual(acf[-1], self.model.ACF(np.inf, w=[0, 1, 0, -1, 0]))

        # If there's no steady state
        msd = self.model.MSD(dts[:-1], w=[0, 0, 1, 0, 0])
        self.assertEqual(msd[0], 0)

        # Various problems that might come up
        with self.assertRaises(ValueError):
            _ = self.model.MSD([-1]) # dt < 0
        with self.assertRaises(ValueError):
            _ = self.model.MSD([np.nan]) # invalid time lag
        with self.assertRaises(ValueError):
            _ = self.model.MSD([np.inf]) # no global steady state
        with self.assertRaises(ValueError):
            _ = self.model.MSD([np.inf], w=[0, 1, 0, 0, 0]) # no steady state

        # With ballistic motion due to force
        self.model.F[3] = [0, 1, 0]
        self.model.update_F_only()
        msd = self.model.MSD(dts[:-1])

        # Some more ACF stuff
        _ = self.model.ACF(dts)
        with self.assertRaises(ValueError):
            _ = self.model.ACF([-1]) # dt < 0
        with self.assertRaises(ValueError):
            _ = self.model.ACF([np.nan]) # invalid time lag

    def test_scales(self):
        ts = self.model.timescales()
        self.assertEqual(ts['t_microscopic'], 1)
        self.assertAlmostEqual(ts['t_equilibration']/ts['t_Rouse'], np.pi**3/4)

        self.assertAlmostEqual(self.model.Gamma(), 6/np.sqrt(np.pi))
        self.assertAlmostEqual(self.model.rms_Ree()**2, 12)
        self.assertAlmostEqual(self.model.rms_Ree(L=3)**2, 9)

class TestMSDfun(myTestCase):
    def test_twolocus(self):
        msd = rouse.twoLocusMSD([0, 1, np.inf], 1, 1)
        self.assertEqual(msd[0], 0)
        self.assertEqual(msd[-1], 2)
        self.assertEqual(rouse.twoLocusMSD(np.inf, 1, 1), 2)

        with self.assertRaises(ValueError):
            _ = rouse.twoLocusMSD([-1], 1, 1)
        with self.assertRaises(ValueError):
            _ = rouse.twoLocusMSD([np.nan], 1, 1)

        # check all time scales
        _ = rouse.twoLocusMSD(np.logspace(-10, 10, 10), 1, 1)

if __name__ == '__main__':
    unittest.main(module=__file__[:-3])
