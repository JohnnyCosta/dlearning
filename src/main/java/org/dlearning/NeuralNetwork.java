package org.dlearning;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.analysis.function.Tanh;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Multi-layer neural network
 *
 * @author : Joao Costa (joaocarlosfilho@gmail.com) on 04/11/2016.
 */
@Slf4j
public final class NeuralNetwork {
    private List<Integer> sizes;

    List<List<List<Double>>> w;

    List<Double> b;

    boolean bias;

    boolean sigmoid;

    public NeuralNetwork(List<Integer> layersSize, List<List<List<Double>>> weight, List<Double> bias, boolean sigmoid) {

        List<List<List<Double>>> wg;
        if (weight == null) {
            wg = genRandomWeights(layersSize);
        } else {
            wg = weight;
        }

        init(layersSize, wg);

        if (bias != null) {
            initBias(bias);
        }

        this.sigmoid = sigmoid;
    }

    private List<List<List<Double>>> genRandomWeights(List<Integer> layersSize) {

        List<List<List<Double>>> w = new ArrayList<>(layersSize.size() - 1);

        Random r = new Random();

        for (int i = 0; i < layersSize.size() - 1; i++) {

            // Remove 4* if not sigmoid
            int nIn = layersSize.get(i);
            int nOut = layersSize.get(i + 1);

            double fanin = -FastMath.sqrt(6. / (nIn + nOut));
            double fanout = FastMath.sqrt(6. / (nIn + nOut));

            if (sigmoid) {
                fanin = 4 * fanin;
                fanout = 4 * fanout;
            }


            List<List<Double>> lw = new ArrayList<>(nOut);

            for (int j = 0; j < nOut; j++) {
                List<Double> nw = new ArrayList<>(nIn);

                for (int k = 0; k < nIn; k++) {
                    nw.add(ThreadLocalRandom.current().nextDouble(fanin, fanout));
                }

                lw.add(nw);
            }

            w.add(lw);
        }

        return w;
    }


    private void initBias(List<Double> bias) {
        assert bias.size() == sizes.size() - 1;
        b = bias;
        this.bias = true;
    }

    public List<List<List<Double>>> getCurrentWeights() {
        return w;
    }

    private void init(List<Integer> layersSize, List<List<List<Double>>> weight) {
        assert layersSize.size() > 2;
        this.sizes = layersSize;

        assert weight.size() == sizes.size() - 1; // Check if weight has N-1 layers, since input layers does not have weights
        int nodeIdx = 0;
        // Check the layers for amount of weights per node
        for (List<List<Double>> layerWeights : weight) {  // for each weight layer
            Integer prevNodes = sizes.get(nodeIdx); // Size of previous layer nodes
            for (List<Double> nodeWeights : layerWeights) {
                assert nodeWeights.size() == prevNodes; // Each weights of a node (for a layer) should be equal to number of previous layer nodes
            }
            nodeIdx++;
        }

        w = weight;

        bias = false;
    }


    public List<List<Double>> calculate(List<Double> x) {
        assert sizes.get(0) == x.size();

        List<List<Double>> out = new ArrayList<>(sizes.size() - 1);

        List<Double> input = x;
        for (int layerIdx = 0; layerIdx < sizes.size() - 1; layerIdx++) {
            input = calculate(input, layerIdx);
            out.add(input);
        }
        return out;
    }

    private List<Double> calculate(List<Double> pout, Integer layerIdx) {
        assert pout.size() == sizes.get(layerIdx);

        List<Double> out = new ArrayList<>(sizes.get(layerIdx + 1));

        for (int j = 0; j < sizes.get(layerIdx + 1); j++) {
            Double oj = 0d;
            for (int k = 0; k < sizes.get(layerIdx); k++) {
                oj += w.get(layerIdx).get(j).get(k) * pout.get(k);
            }

            if (bias) {
                oj += b.get(layerIdx);
            }

            if (sigmoid) {
                out.add(new Sigmoid().value(oj));
            } else {
                out.add(new Tanh().value(oj));
            }
        }
        return out;
    }

    public void train(List<Training<Double>> trainings, double rate, double abort, int maxIter) {

        int iter = 0;
        Double totalerror = 0.d;
        boolean cont = true;
        while (iter < maxIter && cont) {
            totalerror = 0d;

            for (Training t : trainings) {
                assert (t.getInput().size() == sizes.get(0));
                assert (t.getOuput().size() == sizes.get(sizes.size() - 1));

                // Forward pass
                List<List<Double>> output = calculate(t.getInput());

                Double error = 0.d;
                List<Double> outLayer = output.get(output.size() - 1);
                for (int i = 0; i < outLayer.size(); i++) {

                    error += FastMath.pow((Double) t.getOuput().get(i) - outLayer.get(i), 2) / 2;
                }
                error = FastMath.abs(error);
                totalerror += FastMath.abs(error);

                // Backpass
                for (int layer = sizes.size() - 1; layer < 0; layer++) {
                    for (int j = 0; j < sizes.get(layer + 1); j++) {
                        Double deltaj = calculateDelta(j, layer, output, t.getOuput());

                        List<Double> wj = w.get(layer).get(j);
                        for (int i = 0; i < wj.size(); i++) {

                            Double wji = wj.get(i);

                            Double oj = 0.d;
                            if (layer == 0) {
                                oj = (Double) t.getInput().get(j);
                            } else {
                                oj = output.get(layer - 1).get(j);
                            }

                            wji = wji - rate * deltaj * oj;

                            wj.set(i, wji);
                        }
                    }
                }
            }

            if (iter == 0) {
                log.info("Initial total error: '{}'", totalerror);
            }

            if (totalerror.doubleValue() <= abort) {
                cont = false;
            }

            iter++;

        }
        log.info("Finished on iteration : '{}'", iter);
        log.info("Final total error : '{}'", totalerror);
    }

    private Double calculateDelta(int j, int layer, List<List<Double>> layersOut, List<Double> outLayerExpected) {
        assert layer < sizes.size();
        Double oj = (Double) layersOut.get(layer).get(j);
        Double sum = 0d;
        if (layer == sizes.size() - 2) {
            Double tj = (Double) outLayerExpected.get(j);
            sum = (oj - tj);
        } else {
            for (int l = 0; l < sizes.get(layer + 2); l++) {
                sum += calculateDelta(l, layer + 1, layersOut, outLayerExpected) * w.get(layer).get(l).get(j);
            }
        }

        return sum * oj * (1 - oj);
    }
}
