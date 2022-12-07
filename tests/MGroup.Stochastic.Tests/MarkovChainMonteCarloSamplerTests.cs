using System;
using Accord.Statistics;
using Accord.Statistics.Distributions.Multivariate;

using Xunit;

namespace MGroup.Stochastic.Tests
{

	public class MarkovChainMonteCarloSamplerTests
	{
		[Fact]
		public void MetropolisHastingsTest()
		{
			var proposal = new MultivariateNormalDistribution(new double[] { 0 }, new double[,] { { 1 } });
			Func<double[], double> prior = x => 1 / Math.Sqrt(2 * Math.PI) * Math.Exp(-Math.Pow(x[0], 2) / 2);
			Func<double[], double> likelihood = x => 1 / (0.5 * Math.Sqrt(2 * Math.PI)) * Math.Exp(-1 / 2d * Math.Pow((x[0] - 2) / 0.5, 2));
			Func<double[], double> model = x => likelihood(x) * prior(x);
			var sampler = new MetropolisHastings(1, model, proposal);
			var samples = sampler.GenerateSamples(20000);
			var mean = samples.Mean();
			var std = samples.StandardDeviation();
		}

		[Fact]
		public void TransitionalMarkovChainMonteCarloTest()
		{
			var prior = new MultivariateNormalDistribution(new double[] { 0 }, new double[,] { { 1 } });
			var likelihood = new MultivariateNormalDistribution(new double[] { 2 }, new double[,] { { 0.5*0.5 } });
			Func<double[], double> model = x => x[0];
			var sampler = new TransitionalMarkovChainMonteCarlo(1, model, likelihood, prior, scalingFactor: 0.15);
			var samples = sampler.GenerateSamples(50000);
			var mean = samples.Mean();
			var std = samples.StandardDeviation();
		}

		[Fact]
		public void HamiltonianMonteCarloTest()
		{
			var proposal = new MultivariateNormalDistribution(new double[] { 0 }, new double[,] { { 1 } });
			Func<double[], double> prior = x => 1 / Math.Sqrt(2 * Math.PI) * Math.Exp(-Math.Pow(x[0], 2) / 2);
			Func<double[], double> gradPrior = x => - x[0] / Math.Sqrt(2 * Math.PI) * Math.Exp(-Math.Pow(x[0], 2) / 2);
			Func<double[], double> likelihood = x => 1 / (0.5 * Math.Sqrt(2 * Math.PI)) * Math.Exp(-1 / 2d * Math.Pow((x[0] - 2) / 0.5, 2));
			Func<double[], double> gradLikelihood = x => -x[0] / (0.5 * 0.5 * Math.Sqrt(2 * Math.PI)) * Math.Exp(-1 / 2d * Math.Pow((x[0] - 2) / 0.5, 2));
			Func<double[], double> model = x => likelihood(x) * prior(x);
			Func<double[], int, double> gradModel = (x,y) => (gradPrior(x) * likelihood(x)) + (prior(x) * gradLikelihood(x));
			var sampler = new HamiltonianMonteCarlo(1, model, gradModel, proposal);
			var samples = sampler.GenerateSamples(10000);
			var mean = samples.Mean();
			var std = samples.StandardDeviation();
		}
	}
}
