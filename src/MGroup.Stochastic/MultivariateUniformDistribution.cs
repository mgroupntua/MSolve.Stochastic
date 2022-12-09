using System;
using Accord.Math.Random;
using Accord.Statistics.Distributions.Fitting;
using Accord.Statistics.Distributions.Multivariate;
using Accord.Statistics.Distributions;
using Accord;
using Accord.Math;
using Accord.Statistics.Distributions.Sampling;
using System.Reflection.Emit;
using DotNumerics.Optimization;

namespace MGroup.Stochastic
{
	[Serializable]
	public class MultivariateUniformDistribution : MultivariateContinuousDistribution, IFittableDistribution<double[], IFittingOptions>, IFittable<double[], IFittingOptions>, IFittable<double[]>, IFittableDistribution<double[]>, IDistribution<double[]>, IDistribution, ICloneable, ISampleableDistribution<double[]>, IRandomNumberGenerator<double[]>
	{
		private double[] minValue;
		private double[] maxValue;
		private double[] mean;
		private double[] variance;
		private double[,] covariance;

		private double area;


		public MultivariateUniformDistribution(double[] minValue, double[] maxValue) : base(minValue.Length)
		{
			this.minValue = minValue;
			this.maxValue = maxValue;
			this.mean = new double[minValue.Length];
			this.variance = new double[minValue.Length];
			this.covariance = new double[minValue.Length, minValue.Length];
			this.area = 1d;
			for (int i = 0; i < minValue.Length; i++)
			{
				this.mean[i] = (this.maxValue[i] + this.minValue[i]) / 2;
				this.variance[i] = Math.Pow(this.maxValue[i] - this.minValue[i], 2) / 12;
				this.covariance[i, i] = this.variance[i];
				this.area *= (this.maxValue[i] - this.minValue[i]);
			}
		}

		public override double[] Mean => mean;

		public override double[] Variance => variance;

		public override double[,] Covariance => covariance;

		public override object Clone() => new MultivariateUniformDistribution(minValue, maxValue);

		public override void Fit(double[][] observations, double[] weights, IFittingOptions options)
		{
			if (options != null)
			{
				throw new ArgumentException("This method does not accept fitting options.");
			}

			if (weights != null)
			{
				throw new ArgumentException("This distribution does not support weighted samples.");
			}
			double min = double.MaxValue;
			double max = double.MinValue;
			foreach (var observation in observations)
			{
				for (int i = 0; i < observation.Length; i++)
				{
					if (observation[i] < min)
					{
						minValue[i] = observation[i];
					}

					if (observation[i] > max)
					{
						maxValue[i] = observation[i];
					}
				}
			}
		}

		public override double ProbabilityDensityFunction(double[] x)
		{
			return 1.0 / area;
		}

		public override double LogProbabilityDensityFunction(double[] x)
		{
			return 0.0 - System.Math.Log(area);
		}

		public new double[][] Generate(int samples = 1)
		{
			Random random = new Random();
			var totalSamples = new double[samples][];
			for(int i = 0; i < Dimension; i++)
			{
				totalSamples[i] = new double[Dimension];
				for (int j = 0; j < Dimension; j++)
				{
					totalSamples[i][j] = (maxValue[j] - minValue[j]) * random.NextDouble() + minValue[j];
				}
			}
			return totalSamples;
		}

		public override string ToString(string format, IFormatProvider formatProvider)
		{
			return string.Format(formatProvider, "U(x; minValue, maxValue)");
		}
	}
}
