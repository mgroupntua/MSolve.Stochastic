using System;
using Accord.Statistics.Distributions.Multivariate;
using MGroup.LinearAlgebra.Matrices;

namespace MGroup.Stochastic
{
    public class BayesianUpdate
    {
        private MultivariateContinuousDistribution priorDistribution;
        private MultivariateContinuousDistribution likelihoodFunction;
        private Func<double[], double[]> model;
        public IMarkovChainMonteCarloSampler Sampler { get; set; }

        double[] measurementValues;
        double[] measurementError;

        public BayesianUpdate(Func<double[], double[]> model, MultivariateContinuousDistribution priorDistribution, double[] measurementValues, double[] measurementError)
        {
            this.priorDistribution = priorDistribution;
            this.measurementValues = measurementValues;
            this.measurementError = measurementError;
            this.model = model;
            likelihoodFunction = CreateLikelihoodFunction();
        }

        public double PosteriorFunctionEvaluator(double[] input)
        {
            var modelEvaluation = model(input);
            var likelihoodEvaluation = likelihoodFunction.ProbabilityDensityFunction(modelEvaluation);
            var priorEvaluation = priorDistribution.ProbabilityDensityFunction(input);
            return likelihoodEvaluation * priorEvaluation;
        }

        public double LogPosteriorFunctionEvaluator(double[] input)
        {
            var modelEvaluation = model(input);
            var likelihoodEvaluation = likelihoodFunction.LogProbabilityDensityFunction(modelEvaluation);
            var priorEvaluation = priorDistribution.LogProbabilityDensityFunction(input);
            return likelihoodEvaluation + priorEvaluation;
        }

        public double LikelihoodFunctionEvaluator(double[] input)
        {
            var modelEvaluation = model(input);
            var likelihoodEvaluation = likelihoodFunction.ProbabilityDensityFunction(modelEvaluation);
            return likelihoodEvaluation;
        }

        public double LogLikelihoodFunctionEvaluator(double[] input)
        {
            var modelEvaluation = model(input);
            var likelihoodEvaluation = likelihoodFunction.LogProbabilityDensityFunction(modelEvaluation);
            return likelihoodEvaluation;
        }

        public double GradLogLikelihoodFunctionEvaluator(double[] input, int dimension)
        {
            var modelEvaluation = model(input);
            //var inputNext = new double[input.Length];
            //inputNext = inputl
            //inputNext[dimension] += increment;
            //var NextModelEvaluation = model(inputNext);
            //var gradModel = (NextModelEvaluation - modelEvaluation) / increment;
            var invLikelihoodCovarianceMatrix = Matrix.CreateFromArray(likelihoodFunction.Covariance).Invert();
            var invLikelihoodCovariance = invLikelihoodCovarianceMatrix.CopyToArray2D();
            var invPriorCovarianceMatrix = Matrix.CreateFromArray(priorDistribution.Covariance).Invert();
            var invPriorCovariance = invPriorCovarianceMatrix.CopyToArray2D();
            var gradLikelihoodTerm = 0d;
            var gradPriorTerm = 0d;
            for (int j = 0; j < modelEvaluation.Length; j++)
            {
                gradLikelihoodTerm += (modelEvaluation[dimension] - likelihoodFunction.Mean[dimension]) * invLikelihoodCovariance[dimension, j];
                gradPriorTerm += (input[dimension] - priorDistribution.Mean[dimension]) * invPriorCovariance[dimension, j];
            }
            var gradLogLikelihoodEvaluation = gradLikelihoodTerm + gradPriorTerm; // -Math.Log(gradModel * gradTerm * proposalEvaluation * priorEvaluation)

            return gradLogLikelihoodEvaluation;
        }

        public double[] PriorDistributionGenerator()
        {
            var priorSample = priorDistribution.Generate();
            return priorSample;
        }

        public double[,] Solve(int numSamples)
        {
            if (Sampler == null)
                throw new ArgumentException();
            var samples = Sampler.GenerateSamples(numSamples);
            return samples;
        }

		private MultivariateContinuousDistribution CreateLikelihoodFunction()
		{
			var measurementErrors = new double[measurementValues.Length, measurementValues.Length];
			for (int i = 0; i < measurementErrors.GetLength(0); i++)
			{
				measurementErrors[i, i] = Math.Pow(measurementError[i], 2);
			}

			return new MultivariateNormalDistribution(measurementValues, measurementErrors);
		}

	}
}
