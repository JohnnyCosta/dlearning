package org.dlearning;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.util.FastMath;

import java.util.List;

/**
 * Softmax neuron
 *
 * @author : Joao Costa (joaocarlosfilho@gmail.com) on 04/11/2016.
 */
public class SoftmaxNeuron extends GenericPerceptron<Double> {

    //  Weights
    protected List<List<Double>> ws;
    // Bias value
    protected List<Double> bs;
    // Current neuron index
    protected Integer i;


    public SoftmaxNeuron(List<List<Double>> w, List<Double> bias, Integer i) {
        super(w.get(i), bias.get(i), 0d, in -> new Sigmoid().value(in), (a, b) -> a * b, (a, b) -> a + b);
        ws = w;
        bs = bias;
        this.i = i;
    }

    private Double softmax(List<List<Double>> w, List<Double> x, List<Double> bias, Integer i) {
        Double a = FastMath.exp(dotProduct(w.get(i), x) + bias.get(i));

        Double b = 0d;
        for (int j = 0; j < bias.size(); j++) {
            b = b + FastMath.exp(dotProduct(w.get(j), x) + bias.get(j));
        }

        return a / b;
    }

    @Override
    public Double calculate(List<Double> x) {
        return softmax(ws, x, bs, i);
    }
}
