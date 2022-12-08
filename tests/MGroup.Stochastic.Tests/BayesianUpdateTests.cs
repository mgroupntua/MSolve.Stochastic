using System;
using Accord.Statistics;
using Accord.Statistics.Distributions.Multivariate;
using Xunit;

namespace MGroup.Stochastic.Tests
{

	public static class BayesianUpdateTests
	{
		[Fact]
		public static void FiniteElementModelTest()
		{
			var proposal = new MultivariateNormalDistribution(new double[] { 1353000 }, new double[,] { { 135300d * 135300d } });
			var prior = new MultivariateNormalDistribution(new double[] { 1353000 }, new double[,] { { 135300d * 135300d } });
			var measurementValues = new double[] { 2 };
			var measurementError = new double[] { 135300d * 135300d };
			var model = new FiniteElementModel();
			var bayesianInstance = new BayesianUpdate(model.CreateModel, prior, measurementValues, measurementError);
			var sampler = new MetropolisHastings(1, bayesianInstance.PosteriorModel, proposal);
			var samples = sampler.GenerateSamples(1000);
			var mean = samples.Mean();
			var std = samples.StandardDeviation();
			Assert.True(Math.Abs(mean - 1353000) < 50000);
			Assert.True(Math.Abs(std[0] - 135300) < 10000);
		}
	}
}
