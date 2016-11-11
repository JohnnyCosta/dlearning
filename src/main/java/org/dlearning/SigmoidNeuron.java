package org.dlearning;

import org.apache.commons.math3.analysis.function.Sigmoid;

import java.util.List;

/**
 * Sigmoid  function
 *
 * @author : Joao Costa (joaocarlosfilho@gmail.com) on 04/11/2016.
 */
public class SigmoidNeuron implements InputCalculate<Double> {

    private GenericPerceptron<Double> p;

    public SigmoidNeuron(List<Double> weights, Double bias) {

        p = new GenericPerceptron<Double>(weights, bias, 0d, in -> new Sigmoid().value(in), (a, b) -> a * b, (a, b) -> a + b);

    }

    @Override
    public Double calculate(List<Double> x) {
        return p.calculate(x);
    }
}
