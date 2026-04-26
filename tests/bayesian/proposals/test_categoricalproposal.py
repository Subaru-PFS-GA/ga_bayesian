import torch
from unittest import TestCase

from pfs.ga.bayesian import Constants, Defaults
from pfs.ga.bayesian.proposals import CategoricalProposal

class CategoricalProposalTest(TestCase):
    def test_init(self):
        w = torch.tensor([0.2, 0.7, 0.1])
        proposal = CategoricalProposal(None, None, w)
        self.assertEqual(proposal.gamma, Defaults.proposal_gamma)
        self.assertIsNotNone(proposal.w)

    def test_sample(self):
        w = torch.tensor([0.2, 0.7, 0.1])
        proposal = CategoricalProposal(None, None, w)

        s = proposal.sample()
        self.assertEqual(s.shape, ())

        s = proposal.sample(shape=(100,))
        self.assertEqual(s.shape, (100,))

    def test_update(self):
        w = torch.tensor([0.2, 0.7, 0.1])
        proposal = CategoricalProposal(None, None, w)

        s = proposal.sample(shape=(100,))

        # Assume all accepted, so we update with the same samples
        proposal.update(s)