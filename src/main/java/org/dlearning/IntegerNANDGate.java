package org.dlearning;

import java.util.Arrays;
import java.util.List;

/**
 * Create a NAND gate with a perceptron
 *
 * @author : Joao Costa (joaocarlosfilho@gmail.com) on 04/11/2016.
 */
public class IntegerNANDGate implements InputCalculate<Integer>  {

    GenericPerceptron<Integer> p;

    public IntegerNANDGate() {
        p = new GenericPerceptron<Integer>(Arrays.asList(-2, -2), 3, 0, in -> (in <= 0) ? 0:1, (a, b) -> a * b, (a, b) -> a + b);

    }

    @Override
    public Integer calculate(List<Integer> x) {
        return p.calculate(x);
    }

}
