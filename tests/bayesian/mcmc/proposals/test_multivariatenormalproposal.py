import torch
from unittest import TestCase

from pfs.ga.bayesian.mcmc.proposals import MultivariateNormalProposal


class MultivariateNormalProposalTest(TestCase):
	def test_init(self):
		loc = torch.tensor([0.0, 1.0])
		cov = torch.tensor([[1.0, 0.2], [0.2, 2.0]])
		proposal = MultivariateNormalProposal(loc, cov)

		self.assertEqual(proposal.gamma, 0.99)
		self.assertTrue(torch.equal(proposal.loc, loc))
		self.assertTrue(torch.equal(proposal.cov, cov))
		self.assertIsNotNone(proposal.chol)
		self.assertIsNotNone(proposal.eps)

	def test_sample(self):
		loc = torch.tensor([0.0, 1.0])
		cov = torch.tensor([[1.0, 0.2], [0.2, 2.0]])
		proposal = MultivariateNormalProposal(loc, cov)

		sample = proposal.sample()
		self.assertEqual(sample.shape, (2,))

		sample = proposal.sample(shape=(100,))
		self.assertEqual(sample.shape, (100, 2))

	def test_update(self):
		loc = torch.tensor([0.0, 1.0])
		cov = torch.tensor([[1.0, 0.2], [0.2, 2.0]])
		proposal = MultivariateNormalProposal(loc.clone(), cov.clone(), gamma=0.5)

		original_loc = proposal.loc.clone()
		original_chol = proposal.chol.clone()

		proposal.update(torch.tensor([2.0, -1.0]))

		self.assertFalse(torch.equal(proposal.loc, original_loc))
		self.assertFalse(torch.equal(proposal.chol, original_chol))

	def test_update_loc(self):
		proposal = MultivariateNormalProposal(
			torch.tensor([0.0, 1.0]),
			torch.tensor([[1.0, 0.2], [0.2, 2.0]]),
		)

		loc = torch.tensor([0.0, 1.0])
		x = torch.tensor([2.0, -1.0])
		proposal._MultivariateNormalProposal__update_loc(loc, x, gamma=0.5)

		expected = torch.tensor([1.0, 0.0])
		self.assertTrue(torch.allclose(loc, expected))

	def test_update_cov(self):
		proposal = MultivariateNormalProposal(
			torch.tensor([0.0, 1.0]),
			torch.tensor([[1.0, 0.2], [0.2, 2.0]]),
		)

		cov = torch.tensor([[1.0, 0.2], [0.2, 2.0]])
		loc = torch.tensor([0.0, 1.0])
		x = torch.tensor([2.0, -1.0])
		proposal._MultivariateNormalProposal__update_cov(cov, loc, x, gamma=0.5)

		nn = x - loc
		expected = 0.5 * torch.tensor([[1.0, 0.2], [0.2, 2.0]]) + 0.5 * (nn[..., None] * nn[..., None, :])
		self.assertTrue(torch.allclose(cov, expected))

	def test_update_chol(self):
		proposal = MultivariateNormalProposal(
			torch.tensor([0.0, 1.0]),
			torch.tensor([[1.0, 0.2], [0.2, 2.0]]),
		)

		cov = torch.tensor([[1.0, 0.2], [0.2, 2.0]])
		chol = torch.linalg.cholesky(cov)
		loc = torch.tensor([1.0, 0.0])
		x = torch.tensor([2.0, -1.0])
		proposal._MultivariateNormalProposal__update_chol(chol, loc, x, gamma=0.5)

		nn = x - loc
		expected_cov = 0.5 * cov + 0.5 * (nn[..., None] * nn[..., None, :])
		expected_chol = torch.linalg.cholesky(expected_cov)
		self.assertTrue(torch.allclose(chol, expected_chol, atol=1e-6))
