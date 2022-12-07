namespace MGroup.Stochastic
{
    public interface IMarkovChainMonteCarloSampler
    {
        public double[,] GenerateSamples(int numSamples);
    }
}