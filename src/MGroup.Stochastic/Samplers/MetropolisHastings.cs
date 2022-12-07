using Accord.Math.Random;
using Accord.Statistics.Distributions;
using Accord.Statistics.Distributions.Multivariate;
using Accord.Statistics.Distributions.Univariate;
using MGroup.Stochastic.Interfaces;
using System;

namespace MGroup.Stochastic
{
	public class MetropolisHastings : IMarkovChainMonteCarloSampler
	{
		Func<double[], double> model;
		MultivariateNormalDistribution proposalDistribution;

		public double[] initialSample;
		private double[] currentSample;
		private double[] candidateSample;

		private int burnIn;
		private int rejectionRate;
		private int dimensions;
		private Random randomSource;

		public MetropolisHastings(int dimensions, Func<double[], double> model, MultivariateNormalDistribution proposal, int burnIn = 0, int rejectionRate = 0)
		{
			this.model = model;
			this.proposalDistribution = proposal;
			this.burnIn = burnIn;
			this.rejectionRate = rejectionRate;
			this.dimensions = dimensions;
			this.currentSample = new double[dimensions];
			this.candidateSample = new double[dimensions];
			this.randomSource = Generator.Random;
		}

		public double[,] GenerateSamples(int numSamples)
		{
			var samples = new double[numSamples, dimensions];
			var acceptedSamples = 0;
			var currentEvaluation = 0d;
			if (initialSample == null)
				currentEvaluation = Math.Log(model(new double[dimensions]));
			else
				currentEvaluation = Math.Log(model(initialSample));
			while (acceptedSamples < numSamples + burnIn)
			{
				candidateSample = proposalDistribution.Generate();
				var candidateEvaluation = Math.Log(model(candidateSample));
				double Ratio = candidateEvaluation - currentEvaluation;
				if (Math.Log(randomSource.NextDouble()) < Ratio)
				{
					acceptedSamples++;
					if (acceptedSamples > burnIn)
					{
						for (int i = 0; i < dimensions; i++)
						{
							samples[acceptedSamples - burnIn - 1, i] = candidateSample[i];
							currentSample = candidateSample;
							currentEvaluation = candidateEvaluation;
						}
					}
					proposalDistribution = new MultivariateNormalDistribution(candidateSample, proposalDistribution.Covariance);
				}
			}
			return samples;
		}
	}
}
