import torch
from unittest import TestCase

from pfs.ga.bayesian.proposals import NormalProposal


class NormalProposalTest(TestCase):
    def test_init(self):
        proposal = NormalProposal(torch.tensor(1.0), torch.tensor(2.0))

        self.assertEqual(proposal.gamma, 0.99)
        self.assertEqual(proposal.loc.shape, torch.Size([]))
        self.assertEqual(proposal.scale.shape, torch.Size([]))
        self.assertEqual(proposal.eps, 1e-4)

    def test_sample(self):
        proposal = NormalProposal(torch.tensor(0.0), torch.tensor(1.0))

        s = proposal.sample()
        self.assertEqual(s.shape, ())

        s = proposal.sample(shape=(100,))
        self.assertEqual(s.shape, (100,))

    def test_update(self):
        proposal = NormalProposal(torch.tensor(0.0), torch.tensor(1.0), gamma=0.5)

        proposal.update(torch.tensor(2.0))

        self.assertTrue(torch.allclose(proposal.loc, torch.tensor(1.0)))
        self.assertGreater(proposal.scale.item(), 0.0)
