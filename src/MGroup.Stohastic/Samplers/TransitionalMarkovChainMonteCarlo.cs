using System;
using System.Collections.Generic;
using System.Text;
using MGroup.Stochastic.Interfaces;
using Accord.Statistics;
using Accord.Statistics.Distributions;
using Accord.Statistics.Distributions.Multivariate;
using Accord.Statistics.Distributions.Univariate;
using Accord.Math.Random;
using Accord.Math.Decompositions;
using Accord.Math;

namespace MGroup.Stochastic
{
    public class TransitionalMarkovChainMonteCarlo : IMarkovChainMonteCarloSampler
    {
        Func<double[], double> model;
        Func<double[]> initialModel;
        MultivariateNormalDistribution proposalDistribution;

        public double[] initialSample;
        private double[] currentSample;
        private double[] candidateSample;

        private int dimensions;
        private double coefficientOfVariation;
        private double scalingFactor;
        private Random randomSource;

        public TransitionalMarkovChainMonteCarlo(int dimensions, Func<double[], double> model, Func<double[]> initialModel, double coefficientOfVariation = 1, double scalingFactor = 0.2)
        {
            this.model = model;
            this.initialModel = initialModel;
            this.dimensions = dimensions;
            this.coefficientOfVariation = coefficientOfVariation;
            this.scalingFactor = scalingFactor;
            this.currentSample = new double[dimensions];
            this.candidateSample = new double[dimensions];
            this.randomSource = Generator.Random;
        }

        public double[,] GenerateSamples(int numSamples)
        {
            var samples = new double[numSamples, dimensions];
            var modelEvaluations = new double[numSamples];
            for (int i = 0; i < numSamples; i++)
            {
                var priorSample = initialModel();
                for (int j = 0; j < dimensions; j++)
                {
                    samples[i, j] = priorSample[j];
                }
                modelEvaluations[i] = model(priorSample);
            }
            var weights = new double[numSamples];
            var p_current = 0d;
            var b = 0.2d;
            var S = 1d;
            while (p_current < 1)
            {
                var p_temp = 0d;
                var p_lower = p_current;
                var p_upper = 1d;
                while (p_upper - p_lower > 0.0001)
                {
                    p_temp = (p_lower + p_upper) / 2;
                    for (int i = 0; i < numSamples; i++)
                    {
                        weights[i] = Math.Pow(modelEvaluations[i], p_temp - p_current);
                    }
                    var currentCoV = Measures.StandardDeviation(weights) / Measures.Mean(weights);
                    if (currentCoV > coefficientOfVariation)
                    {
                        p_upper = p_temp;
                    }
                    else
                    {
                        p_lower = p_temp;
                    }
                }
                var sumWeights = weights.Sum();
                p_current = p_temp;
                if (1 - p_current < 0.001)
                {
                    p_current = 1;
                }
                var normalizedWeights = new double[numSamples];
                for (int i = 0; i < numSamples; i++)
                {
                    normalizedWeights[i] = weights[i] / sumWeights;
                }
                var meanProposal = new double[dimensions];
                var covarProposal = new double[dimensions, dimensions];
                for (int j = 0; j < dimensions; j++)
                {
                    for (int i = 0; i < numSamples; i++)
                    {
                        meanProposal[j] += normalizedWeights[i] * samples[i,j];
                    }
                }
                for (int ii = 0; ii < dimensions; ii++)
                    for (int jj = 0; jj < dimensions; jj++)
                        for (int i = 0; i < numSamples; i++)
                        {
                            covarProposal[ii, jj] += b * b * normalizedWeights[i] * (samples[i, ii] - meanProposal[ii]) * (samples[i, jj] - meanProposal[jj]);
                        }
                int[] randomSampleInd = GeneralDiscreteDistribution.Random(normalizedWeights, numSamples);
                var currentSamples = (double[,])samples.Clone();
                var currentmodelEvaluations = (double[])modelEvaluations.Clone();
                for (int i = 0; i < numSamples; i++)
                {
                    double[][] candidateSample = MultivariateNormalDistribution.Generate(samples: 1, mean: currentSamples.GetRow<double>(randomSampleInd[i]), covariance: covarProposal);
                    var candidateModelEvaluation = model(candidateSample[0]);
                    var modelEvaluationRatio = Math.Pow(candidateModelEvaluation / modelEvaluations[i], p_current);
                    if (modelEvaluationRatio > Generator.Random.NextDouble())
                    {
                        for (int j = 0; j < dimensions; j++)
                        {
                            samples[i,j] = candidateSample[0][j];
                            currentSamples[randomSampleInd[i], j] = candidateSample[0][j];
                        }
                        modelEvaluations[i] = candidateModelEvaluation;
                        currentmodelEvaluations[randomSampleInd[i]] = candidateModelEvaluation;
                    }
                    else
                    {
                        for (int j = 0; j < dimensions; j++)
                        {
                            samples[i, j] = currentSamples[randomSampleInd[i],j];
                        }
                        modelEvaluations[i] = currentmodelEvaluations[randomSampleInd[i]];
                    }
                }
            }
            return samples;
        }
    }
}
