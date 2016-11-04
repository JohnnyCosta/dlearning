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
        p = new GenericPerceptron<Integer>(Arrays.asList(-2, -2), 3, 0, in -> {
            if (in <= 0) {
                return 0;
            } else {
                return 1;
            }
        }, (v1, v2) -> {
            return v1 * v2;
        }, (v1, v2) -> {
            return v1 + v2;
        });

    }

    @Override
    public Integer calculate(List<Integer> x) {
        return p.calculate(x);
    }

}
