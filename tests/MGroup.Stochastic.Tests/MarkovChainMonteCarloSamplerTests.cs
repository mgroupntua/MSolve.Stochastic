using System;
using Accord.Statistics;
using Accord.Statistics.Distributions.Multivariate;
using Xunit;

namespace MGroup.Stochastic.Tests
{

	public static class MarkovChainMonteCarloSamplerTests
	{
		[Fact]
		public static void MetropolisHastingsTest()
		{
			var proposal = new MultivariateNormalDistribution(new double[] { 0 }, new double[,] { { 1 } });
			Func<double[], double> prior = x => new MultivariateNormalDistribution(new double[] { 0 }, new double[,] { { 1 } }).LogProbabilityDensityFunction(x[0]);
			Func<double[], double> likelihood = x => new MultivariateNormalDistribution(new double[] { 2 }, new double[,] { { 0.5 * 0.5 } }).LogProbabilityDensityFunction(x[0]);
			Func<double[], double> model = x => likelihood(x) + prior(x);
			var sampler = new MetropolisHastings(1, model, proposal);
			var samples = sampler.GenerateSamples(10000);
			var mean = samples.Mean();
			var std = samples.StandardDeviation();
			Assert.True(Math.Abs(mean - 1.6) < 0.05);
			Assert.True(Math.Abs(std[0] - 0.45) < 0.1);
		}

		[Fact]
		public static void TransitionalMarkovChainMonteCarloTest()
		{
			var prior = new MultivariateNormalDistribution(new double[] { 0 }, new double[,] { { 1 } });
			Func<double[], double> model = x => new MultivariateNormalDistribution(new double[] { 2 }, new double[,] { { 0.5 * 0.5 } }).LogProbabilityDensityFunction(x[0]);
			var sampler = new TransitionalMarkovChainMonteCarlo(1, model, prior, scalingFactor: 0.2);
			var samples = sampler.GenerateSamples(10000);
			var mean = samples.Mean();
			var std = samples.StandardDeviation();
			Assert.True(Math.Abs(mean - 1.6) < 0.05);
			Assert.True(Math.Abs(std[0] - 0.45) < 0.1);
		}

		[Fact]
		public static void HamiltonianMonteCarloTest()
		{
			var proposal = new MultivariateNormalDistribution(new double[] { 0 }, new double[,] { { 1 } });
			Func<double[], double> prior = x => new MultivariateNormalDistribution(new double[] { 0 }, new double[,] { { 1 } }).LogProbabilityDensityFunction(x[0]);
			Func<double[], double> likelihood = x => new MultivariateNormalDistribution(new double[] { 2 }, new double[,] { { 0.5 * 0.5 } }).LogProbabilityDensityFunction(x[0]);
			Func<double[], double> model = x => likelihood(x) + prior(x);
			Func<double[], double> gradPrior = x => -x[0];
			Func<double[], double> gradLikelihood = x => -2 * (2 * x[0] - 4);
			Func<double[], int, double> gradModel = (x,y) => gradPrior(x) + gradLikelihood(x);
			var sampler = new HamiltonianMonteCarlo(1, model, gradModel, proposal);
			var samples = sampler.GenerateSamples(10000);
			var mean = samples.Mean();
			var std = samples.StandardDeviation();
			Assert.True(Math.Abs(mean - 1.6) < 0.05);
			Assert.True(Math.Abs(std[0] - 0.45) < 0.1);
		}
	}
}
