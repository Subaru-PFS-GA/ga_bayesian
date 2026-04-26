import torch
from unittest import TestCase

from pfs.ga.bayesian import Constants, Defaults
from pfs.ga.bayesian.proposals import DirichletProposal


class DirichletProposalTest(TestCase):
	def test_init(self):
		alpha = torch.tensor([2.0, 3.0, 5.0])
		proposal = DirichletProposal(None, None, alpha)

		self.assertEqual(proposal.gamma, Defaults.proposal_gamma)
		self.assertTrue(torch.equal(proposal.alpha, alpha))
		self.assertEqual(proposal.m, Defaults.proposal_dirichlet_max_concentration)

	def test_sample(self):
		alpha = torch.tensor([2.0, 3.0, 5.0])
		proposal = DirichletProposal(None, None, alpha)

		sample = proposal.sample()
		self.assertEqual(sample.shape, (3,))

		sample = proposal.sample(shape=(100,))
		self.assertEqual(sample.shape, (100, 3))

	def test_update(self):
		alpha = torch.tensor([2.0, 3.0, 5.0])
		proposal = DirichletProposal(None, None, alpha, m=4.0, gamma=0.5)

		proposal.update(torch.tensor([0.7, 0.2, 0.1]))

		self.assertFalse(torch.equal(proposal.alpha, alpha))
		self.assertTrue(torch.all(proposal.alpha > 0))
		self.assertLessEqual(proposal.alpha.max().item(), 4.0)
