using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class FeedForwardNeuralNetwork
    {
        public static List<double[]> inputList = new List<double[]>() {
            new double[] { 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1 }
        };

        public static List<double[]> outputList = new List<double[]>() {
            new double[] { 1, 0, 0, 0 },
            new double[] { 0, 1, 0, 0 },
            new double[] { 0, 0, 1, 0 },
            new double[] { 0, 0, 0, 1 }
        };

        static void Main(String[] args)
        {
            Random random = new Random();
            double[,] weights1 = new double[12, 12];

            double[,] weights2 = new double[10, 12];

            double[,] weights3 = new double[8, 10];

            double[,] weights4 = new double[4, 8];

            double[] bias1 = new double[12];

            double[] bias2 = new double[10];

            double[] bias3 = new double[8];

            double[] bias4 = new double[4];

            for (int i = 0; i < 12; i++)
            {
                for (int j = 0; j < 12; j++)
                {
                    weights1[i, j] = Sigmoid(random.Next(0, 100));
                }
            }

            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 12; j++)
                {
                    weights2[i, j] = Sigmoid(random.Next(0, 100));
                }
            }

            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    weights3[i, j] = Sigmoid(random.Next(0, 100));
                }
            }

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    weights4[i, j] = Sigmoid(random.Next(0, 100));
                }
            }

            for (int i = 0; i < 12; i++)
            {
                bias1[i] = Sigmoid(random.Next(0, 100));
            }

            for (int i = 0; i < 10; i++)
            {
                bias2[i] = Sigmoid(random.Next(0, 100)); 
            }

            for (int i = 0; i < 8; i++)
            {
                bias3[i] = Sigmoid(random.Next(0, 100));
            }

            for (int i = 0; i < 4; i++)
            {
                bias4[i] = Sigmoid(random.Next(0, 100));
            }

            double[,] holderw1 = new double[12, 12];
            double[,] holderw2 = new double[10, 12];
            double[,] holderw3 = new double[8, 10];
            double[,] holderw4 = new double[4, 8];
            double[] holderb1 = new double[12];
            double[] holderb2 = new double[10];
            double[] holderb3 = new double[8];
            double[] holderb4 = new double[4];

            //training
            double lr = 4.5;
            for (int u = 0; u < 10000; u++)
            {
                foreach ((double[] inpSet, double[] outpSet) in inputList.Zip(outputList, (x1, x2) => (x1, x2)))
                {
                    for (int l = 0; l < 100; l++)
                    {
                        if (l % 10 == 0 && l != 0)
                            Console.WriteLine("Cost on this nth iteration on this set is " + Cost(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet));
                        //optimizer for weights1
                        for (int i = 0; i < weights1.GetLength(0); i++)
                        {
                            for (int j = 0; j < weights1.GetLength(1); j++)
                            {
                                double partial = CostPDerivw1(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet, i, j);
                                holderw1[i, j] = weights1[i, j] - (lr * partial);
                            }
                        }
                        //optimizer for weights2
                        for (int i = 0; i < weights2.GetLength(0); i++)
                        {
                            for (int j = 0; j < weights2.GetLength(1); j++)
                            {
                                double partial = CostPDerivw2(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet, i, j);
                                holderw2[i, j] = weights2[i, j] - (lr * partial);
                            }
                        }
                        //optimizer for weights3
                        for (int i = 0; i < weights3.GetLength(0); i++)
                        {
                            for (int j = 0; j < weights3.GetLength(1); j++)
                            {
                                double partial = CostPDerivw3(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet, i, j);                               
                                holderw3[i, j] = weights3[i, j] - (lr * partial);
                            }
                        }
                        //optimizer for weights4
                        for (int i = 0; i < weights4.GetLength(0); i++)
                        {
                            for (int j = 0; j < weights4.GetLength(1); j++)
                            {
                                double partial = CostPDerivw4(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet, i, j);                           
                                holderw4[i, j] = weights4[i, j] - (lr * partial);
                            }
                        }
                        //optimizer for bias1
                        for (int i = 0; i < bias1.Length; i++)
                        {
                            double partial = CostPDerivb1(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet, i);
                            holderb1[i] = bias1[i] - (lr * partial);
                        }
                        //optimizer for bias2
                        for (int i = 0; i < bias2.Length; i++)
                        {
                            double partial = CostPDerivb2(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet, i);
                            holderb2[i] = bias2[i] - (lr * partial);
                        }
                        //optimizer for bias3
                        for (int i = 0; i < bias3.Length; i++)
                        {
                            double partial = CostPDerivb3(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet, i);
                            holderb3[i] = bias3[i] - (lr * partial);
                        }
                        //optimizer for bias4
                        for (int i = 0; i < bias4.Length; i++)
                        {
                            double partial = CostPDerivb4(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet, i);
                            holderb4[i] = bias4[i] - (lr * partial);
                        }

                        //updating weights1
                        for (int i = 0; i < weights1.GetLength(0); i++)
                        {
                            for (int j = 0; j < weights1.GetLength(1); j++)
                            {
                                weights1[i, j] = holderw1[i, j];
                            }
                        }
                        //updating weights2
                        for (int i = 0; i < weights2.GetLength(0); i++)
                        {
                            for (int j = 0; j < weights2.GetLength(1); j++)
                            {
                                weights2[i, j] = holderw2[i, j];
                            }
                        }
                        //updating weights3
                        for (int i = 0; i < weights3.GetLength(0); i++)
                        {
                            for (int j = 0; j < weights3.GetLength(1); j++)
                            {
                                weights3[i, j] = holderw3[i, j];
                            }
                        }
                        //updating weights4
                        for (int i = 0; i < weights4.GetLength(0); i++)
                        {
                            for (int j = 0; j < weights4.GetLength(1); j++)
                            {
                                weights4[i, j] = holderw4[i, j];
                            }
                        }
                        //updating bias1
                        for (int i = 0; i < bias1.Length; i++)
                        {
                            bias1[i] = holderb1[i];
                        }
                        //updating bias2
                        for (int i = 0; i < bias2.Length; i++)
                        {
                            bias2[i] = holderb2[i];
                        }
                        //updating bias3
                        for (int i = 0; i < bias3.Length; i++)
                        {
                            bias3[i] = holderb3[i];
                        }
                        //updating bias4
                        for (int i = 0; i < bias4.Length; i++)
                        {
                            bias4[i] = holderb4[i];
                        }

                        lr *= 0.985;
                    }
                    lr = 4.5;
                }
                lr = 4.5;
            }
        }

        public static double Cost(double[,] weights1, double[,]weights2, double[,] weights3, double[,] weights4, double[] bias1, double[] bias2, double[] bias3, double[] bias4, double[] inpSet, double[] outpSet)
        {
            double[] hLayer1 = Sigmoid(AddVec(bias1, MultVecMa(weights1, inpSet)));
            double[] hLayer2 = Sigmoid(AddVec(bias2, MultVecMa(weights2, hLayer1)));
            double[] hLayer3 = Sigmoid(AddVec(bias3, MultVecMa(weights3, hLayer2)));
            double[] outLayer = Sigmoid(AddVec(bias4, MultVecMa(weights4, hLayer3)));

            double[] errVec = SubVec(outLayer, outpSet);
            double error = 0;
            for (int i = 0; i < errVec.Length; i++)
            {
                error += 0.5 * Math.Pow(errVec[i], 2);
            }
            return error;
        }

        public static double CostPDerivw1(double[,] weights1, double[,] weights2, double[,] weights3, double[,] weights4, double[] bias1, double[] bias2, double[] bias3, double[] bias4, double[] inpSet, double[] outpSet, int pos1, int pos2)
        {
            double r = 0.0001;
            double[,] weights1n = new double[weights1.GetLength(0), weights1.GetLength(1)];
            for (int i = 0; i < weights1.GetLength(0); i++)
            {
                for (int j = 0; j < weights1.GetLength(1); j++)
                {
                    weights1n[i, j] = weights1[i, j];
                }
            }
            weights1n[pos1, pos2] += r;

            double partial = (Cost(weights1n, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet) - Cost(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet)) / r;
            return partial;
        }

        public static double CostPDerivw2(double[,] weights1, double[,] weights2, double[,] weights3, double[,] weights4, double[] bias1, double[] bias2, double[] bias3, double[] bias4, double[] inpSet, double[] outpSet, int pos1, int pos2)
        {
            double r = 0.0001;
            double[,] weights2n = new double[weights2.GetLength(0), weights2.GetLength(1)];
            for (int i = 0; i < weights2.GetLength(0); i++)
            {
                for (int j = 0; j < weights2.GetLength(1); j++)
                {
                    weights2n[i, j] = weights2[i, j];
                }
            }
            weights2n[pos1, pos2] += r;

            double partial = (Cost(weights1, weights2n, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet) - Cost(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet)) / r;
            return partial;
        }

        public static double CostPDerivw3(double[,] weights1, double[,] weights2, double[,] weights3, double[,] weights4, double[] bias1, double[] bias2, double[] bias3, double[] bias4, double[] inpSet, double[] outpSet, int pos1, int pos2)
        {
            double r = 0.0001;
            double[,] weights3n = new double[weights3.GetLength(0), weights3.GetLength(1)];
            for (int i = 0; i < weights3.GetLength(0); i++)
            {
                for (int j = 0; j < weights3.GetLength(1); j++)
                {
                    weights3n[i, j] = weights3[i, j];
                }
            }
            weights3n[pos1, pos2] += r;

            double partial = (Cost(weights1, weights2, weights3n, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet) - Cost(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet)) / r;
            return partial;
        }

        public static double CostPDerivw4(double[,] weights1, double[,] weights2, double[,] weights3, double[,] weights4, double[] bias1, double[] bias2, double[] bias3, double[] bias4, double[] inpSet, double[] outpSet, int pos1, int pos2)
        {
            double r = 0.0001;
            double[,] weights4n = new double[weights4.GetLength(0), weights4.GetLength(1)];
            for (int i = 0; i < weights4.GetLength(0); i++)
            {
                for (int j = 0; j < weights4.GetLength(1); j++)
                {
                    weights4n[i, j] = weights4[i, j];
                }
            }
            weights4n[pos1, pos2] += r;

            double partial = (Cost(weights1, weights2, weights3, weights4n, bias1, bias2, bias3, bias4, inpSet, outpSet) - Cost(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet)) / r;
            return partial;
        }
        
        public static double CostPDerivb1(double[,] weights1, double[,] weights2, double[,] weights3, double[,] weights4, double[] bias1, double[] bias2, double[] bias3, double[] bias4, double[] inpSet, double[] outpSet, int pos)
        {
            double r = 0.0001;
            double[] bias1n = new double[bias1.Length];
            for (int i = 0; i < bias1.Length; i++)
            {
                bias1n[i] = bias1[i];
            }
            bias1n[pos] += r;

            double partial = (Cost(weights1, weights2, weights3, weights4, bias1n, bias2, bias3, bias4, inpSet, outpSet) - Cost(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet)) / r;
            return partial;          
        }

        public static double CostPDerivb2(double[,] weights1, double[,] weights2, double[,] weights3, double[,] weights4, double[] bias1, double[] bias2, double[] bias3, double[] bias4, double[] inpSet, double[] outpSet, int pos)
        {
            double r = 0.0001;
            double[] bias2n = new double[bias2.Length];
            for (int i = 0; i < bias2.Length; i++)
            {
                bias2n[i] = bias2[i];
            }
            bias2n[pos] += r;

            double partial = (Cost(weights1, weights2, weights3, weights4, bias1, bias2n, bias3, bias4, inpSet, outpSet) - Cost(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet)) / r;
            return partial;
        }

        public static double CostPDerivb3(double[,] weights1, double[,] weights2, double[,] weights3, double[,] weights4, double[] bias1, double[] bias2, double[] bias3, double[] bias4, double[] inpSet, double[] outpSet, int pos)
        {
            double r = 0.0001;
            double[] bias3n = new double[bias3.Length];
            for (int i = 0; i < bias3.Length; i++)
            {
                bias3n[i] = bias3[i];
            }
            bias3n[pos] += r;

            double partial = (Cost(weights1, weights2, weights3, weights4, bias1, bias2, bias3n, bias4, inpSet, outpSet) - Cost(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet)) / r;
            return partial;
        }

        public static double CostPDerivb4(double[,] weights1, double[,] weights2, double[,] weights3, double[,] weights4, double[] bias1, double[] bias2, double[] bias3, double[] bias4, double[] inpSet, double[] outpSet, int pos)
        {
            double r = 0.0001;
            double[] bias4n = new double[bias4.Length];
            for (int i = 0; i < bias4.Length; i++)
            {
                bias4n[i] = bias4[i];
            }
            bias4n[pos] += r;

            double partial = (Cost(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4n, inpSet, outpSet) - Cost(weights1, weights2, weights3, weights4, bias1, bias2, bias3, bias4, inpSet, outpSet)) / r;
            return partial;
        }

        public static double Sigmoid(double x)
        {
            double final = 1 / (1 + Math.Exp(-1 * x));
            return final;
        }

        public static double[] Sigmoid(double[] xvec)
        {
            double[] final = new double[xvec.Length];
            for (int i = 0; i < xvec.Length; i++)
            {
                final[i] = 1 / (1 + Math.Exp(-1 * xvec[i]));
            }
            return final;
        }

        public static double[] MultVecMa(double[,] matrix, double[] vector)
        {
            double[] final = new double[matrix.GetLength(0)];

            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                final[i] = 0;
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    final[i] += matrix[i, j] * vector[j];
                }
            }

            return final;
        }

        public static double[] AddVec(double[] A, double[] B)
        {
            double[] final = new double[A.Length];
            for (int i = 0; i < A.Length; i++)
            {
                final[i] = A[i] + B[i];
            }
            return final;
        }

        public static double[] SubVec(double[] A, double[] B)
        {
            double[] final = new double[A.Length];
            for (int i = 0; i < A.Length; i++)
            {
                final[i] = A[i] - B[i];
            }
            return final;
        }
    }
}
