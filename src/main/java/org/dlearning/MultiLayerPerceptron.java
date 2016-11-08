package org.dlearning;

import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Sigmoid  neuron based on a perceptron
 *
 * @author : Joao Costa (joaocarlosfilho@gmail.com) on 04/11/2016.
 */
@Slf4j
public final class MultiLayerPerceptron {
    private List<List<InputCalculate<Double>>> layers;
    private List<Integer> sizes;

    public MultiLayerPerceptron(List<Integer> layersSize, List<List<List<Double>>> weight, List<List<Double>> bias) {
        assert layersSize.size() > 2;
        this.sizes = layersSize;

        assert weight.size() == sizes.size() - 1;
        int nodeIdx = 0;
        for (List<List<Double>> layerWeights : weight) { // Check the layers for amount of weights per node
            Integer prevNodes = sizes.get(nodeIdx);
            for (List<Double> nodeWeights : layerWeights) {
                assert nodeWeights.size() == prevNodes;
            }
            nodeIdx++;
        }

        assert bias.size() == sizes.size() - 1;
        nodeIdx = 1;
        for (List<Double> b : bias) { // Check the layers for bias
            assert b.size() == sizes.get(nodeIdx);
            nodeIdx++;
        }

        // Create hidden and output layers neurons
        // TODO review this
        this.layers = new ArrayList<>();
        for (int layerIdx = 0; layerIdx < sizes.size() - 1; layerIdx++) {
            Integer currentNum = sizes.get(layerIdx + 1);
            Integer prevNum = sizes.get(layerIdx);
            List<InputCalculate<Double>> nodes = new ArrayList<>();

            for (int row = 0; row < currentNum; row++) {
                List<Double> nodeWeights = new ArrayList<>(prevNum);
                List<List<Double>> layerWeights = weight.get(layerIdx);


                if (layerIdx < sizes.size() - 2) {
                    nodes.add(new SigmoidNeuron(layerWeights.get(row), bias.get(layerIdx).get(row)));
                } else {
                    nodes.add(new SoftmaxNeuron(layerWeights, bias.get(layerIdx), row));
                }
            }
            layers.add(nodes);
        }
    }

    public List<Double> calculate(List<Double> x) {
        assert sizes.get(0) == x.size();

        List<Double> input = x;
        for (int layerIdx = 0; layerIdx < layers.size(); layerIdx++) {
            input = calculate(input, layerIdx);
            log.info("Output of layer '{}' is '{}'", layerIdx, input);
        }
        return input;
    }

    private List<Double> calculate(List<Double> x, Integer layerIdx) {
        assert x.size() == sizes.get(layerIdx);
        List<InputCalculate<Double>> layer = layers.get(layerIdx);

        assert layer.size() == sizes.get(layerIdx + 1);
        List<Double> out = new ArrayList<>(sizes.get(layerIdx + 1));

        for (InputCalculate<Double> n : layer) {
            out.add(n.calculate(x));
        }

        return out;

    }
}
