import torch
from unittest import TestCase

from pfs.ga.bayesian.distributions import Delta

class DeltaTest(TestCase):
    def test_init(self):
        loc = torch.tensor([0.0, 0.0, 0.0])
        d = Delta(loc)
        self.assertEqual(d.loc.shape, (3,))
        self.assertEqual(d.batch_shape, ())
        self.assertEqual(d.event_shape, (3,))

    def test_sample(self):
        loc = torch.tensor([0.0, 0.0, 0.0])
        d = Delta(loc)

        s = d.sample()
        self.assertEqual(s.shape, (3,))

        s = d.sample(sample_shape=(100,))
        self.assertEqual(s.shape, (100, 3))

        s = d.sample(sample_shape=(100, 5))
        self.assertEqual(s.shape, (100, 5, 3))

    def test_expand(self):
        loc = torch.tensor([0.0, 0.0, 0.0])
        d = Delta(loc)

        self.assertEqual(d.batch_shape, ())
        self.assertEqual(d.event_shape, (3,))

        d_expanded = d.expand((10,))
        self.assertEqual(d_expanded.batch_shape, (10,))
        self.assertEqual(d_expanded.event_shape, (3,))

        s = d_expanded.sample()
        self.assertEqual(s.shape, (10, 3))

        s = d_expanded.sample(sample_shape=(100,))
        self.assertEqual(s.shape, (100, 10, 3))

    def test_log_prob(self):
        loc = torch.tensor([0.0, 0.0, 0.0])
        d = Delta(loc)

        s = d.sample()
        lp = d.log_prob(s)
        self.assertEqual(lp.shape, ())

        s = d.sample(sample_shape=(100,))
        lp = d.log_prob(s)
        self.assertEqual(lp.shape, (100,))

        s = d.sample(sample_shape=(100, 5))
        lp = d.log_prob(s)
        self.assertEqual(lp.shape, (100, 5))