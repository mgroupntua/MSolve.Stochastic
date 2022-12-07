using System;
using Accord.Statistics.Distributions.Multivariate;
using Accord.Math.Random;
using MGroup.LinearAlgebra.Matrices;

namespace MGroup.Stochastic
{
    public class HamiltonianMonteCarlo : IMarkovChainMonteCarloSampler
    {
        Func<double[], double> model;
        Func<double[], int, double> gradModel;
        Func<double[]> initialModel;
        MultivariateNormalDistribution proposalDistribution;

        public double[] initialSample;
        private double[] currentSample;
        private double[] candidateSample;

        private int dimensions;
        private int numSteps;
        private double stepSize;
        private Random randomSource;

        public HamiltonianMonteCarlo(int dimensions, Func<double[], double> model, Func<double[], int, double> gradModel, MultivariateNormalDistribution proposal, int numSteps = 25, double stepSize = 0.25)
        {
            this.model = model;
            this.gradModel = gradModel;
            this.proposalDistribution = proposal;
            this.dimensions = dimensions;
            this.numSteps = numSteps;
            this.stepSize = stepSize;
            this.currentSample = new double[dimensions];
            this.candidateSample = new double[dimensions];
            this.randomSource = Generator.Random;
        }

        public double[,] GenerateSamples(int numSamples)
        {
            var samples = new double[numSamples, dimensions];
            var acceptedSamples = 0;
            var current_q = new double[dimensions];
            var current_gradModel = 0d;
            var current_gradProposal = 0d;
            var current_U = -Math.Log(model(current_q));
            var current_K = 0d;
            var proposed_U = 0d;
            var proposed_K = 0d;

            while (acceptedSamples < numSamples)
            {
                var q = (double[])current_q.Clone();
                var p = proposalDistribution.Generate();
                var current_p = p.Copy();
                for (int k = 0; k < dimensions; k++)
                {
                    current_gradModel = gradModel(q, k);
                    p[k] = p[k] - stepSize * current_gradModel / 2;
                    for (int j = 0; j < numSteps; j++)
                    {
                        current_gradProposal = CalculateGradMinusLogProposal(p, k);
                        q[k] = q[k] + stepSize * current_gradProposal;
                        if (j + 1 != numSteps)
                        {
                            current_gradModel = gradModel(q, k);
                            p[k] = p[k] - stepSize * current_gradModel;
                        }
                    }
                    current_gradModel = gradModel(q, k);
                    p[k] = p[k] - stepSize * current_gradModel / 2;
                    p[k] = -p[k];
                }
                current_K = CalculateMinusLogProposal(current_p);
                proposed_U = -Math.Log(model(q));
                proposed_K = CalculateMinusLogProposal(p);

                if (Math.Log(randomSource.NextDouble()) < Math.Exp (current_U - proposed_U + current_K - proposed_K))
                {
                    acceptedSamples++;
                    for (int i = 0; i < dimensions; i++)
                    {
                        samples[acceptedSamples - 1, i] = q[i];
                        current_q[i] = q[i];
                        current_U = -Math.Log(model(current_q));
                    }
                }
            }
            return samples;
        }

        private double CalculateMinusLogProposal(double[] input)
        {
            var proposalEvaluation = -proposalDistribution.LogProbabilityDensityFunction(input);
            return proposalEvaluation;
        }
        
        private double CalculateGradMinusLogProposal(double[] input, int dimension)
        {
            var invCovarianceMatrix = Matrix.CreateFromArray(proposalDistribution.Covariance).Invert();
            var invCovariance = invCovarianceMatrix.CopyToArray2D();
            var gradTerm = 0d;
            for (int j = 0; j < input.Length; j++)
            {
                gradTerm += (input[dimension] - proposalDistribution.Mean[dimension]) * invCovariance[dimension, j];
            }
            var gradProposalEvaluation = gradTerm;

            return gradProposalEvaluation;
        }
    }
}
