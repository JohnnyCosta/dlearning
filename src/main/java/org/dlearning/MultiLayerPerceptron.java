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

        assert bias.size() == sizes.size() - 1; // Check if bias size to be N-1 layers
        nodeIdx = 1;
        // Check each layers for bias size
        for (List<Double> b : bias) {
            assert b.size() == sizes.get(nodeIdx);
            nodeIdx++;
        }

        // Create hidden and output layers neurons
        this.layers = new ArrayList<>();
        for (int hioutLayerIdx = 0; hioutLayerIdx < sizes.size() - 1; hioutLayerIdx++) { // loop for output and hidden layer only
            Integer currentNum = sizes.get(hioutLayerIdx + 1);
            Integer prevNum = sizes.get(hioutLayerIdx);
            List<InputCalculate<Double>> nodes = new ArrayList<>();

            for (int row = 0; row < currentNum; row++) {
                List<Double> nodeWeights = new ArrayList<>(prevNum); // Each neuron should have be connected to previous nodes size (weight)
                List<List<Double>> layerWeights = weight.get(hioutLayerIdx);


//                if (hioutLayerIdx < sizes.size() - 2) {
                    nodes.add(new SigmoidNeuron(layerWeights.get(row), bias.get(hioutLayerIdx).get(row)));
//                } else { // Output layer with a softmax with need all layer weights and bias
//                    nodes.add(new SoftmaxNeuron(layerWeights, bias.get(hioutLayerIdx), row));
//                }
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
