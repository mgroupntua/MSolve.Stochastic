//using MGroup.Constitutive.Structural;
//using MGroup.Constitutive.Structural.BoundaryConditions;
//using MGroup.FEM.Structural.Line;
//using MGroup.MSolve.Discretization.Entities;
//using MGroup.Stochastic.Interfaces;
//using MGroup.Stochastic.Structural.StochasticRealizers;

//namespace MGroup.Stochastic.Structural
//{
//    public class ModelBuilder
//    {
//        public Model GetModel(RandomVariable randomVariable, IStochasticDomainMapper domainMapper, int iteration)
//        {
//            var model = new Model();
//            model.NodesDictionary.Add(0, new Node(id: 0, x: 0, y: 0, z: 0));
//            model.NodesDictionary.Add(1, new Node(id: 1, x: 1, y: 0, z: 0));
//            model.ElementsDictionary.Add(1, new EulerBeam3D(randomVariable.Realize(iteration, domainMapper, null), 0.3)
//            {
//                ID = 1,
//            });

//			model.BoundaryConditions.Add(new StructuralBoundaryConditionSet(
//				new[]
//				{
//					new NodalDisplacement(model.NodesDictionary[1], StructuralDof.TranslationX, amount: 0d),
//					new NodalDisplacement(model.NodesDictionary[1], StructuralDof.TranslationY, amount: 0d),
//					new NodalDisplacement(model.NodesDictionary[1], StructuralDof.TranslationZ, amount: 0d),
//					new NodalDisplacement(model.NodesDictionary[1], StructuralDof.RotationX, amount: 0d),
//					new NodalDisplacement(model.NodesDictionary[1], StructuralDof.RotationY, amount: 0d),
//					new NodalDisplacement(model.NodesDictionary[1], StructuralDof.RotationZ, amount: 0d)
//				},
//				new[]
//				{
//					new NodalLoad(model.NodesDictionary[1], StructuralDof.TranslationX, amount: 10d)
//				}
//			));

//			return model;
//        }
//    }
//}
