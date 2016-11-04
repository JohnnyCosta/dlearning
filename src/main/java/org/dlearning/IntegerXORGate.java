package org.dlearning;

import java.util.Arrays;
import java.util.List;

/**
 * Create a XOR gate with a perceptron
 *
 * @author : Joao Costa (joaocarlosfilho@gmail.com) on 04/11/2016.
 */
public class IntegerXORGate implements InputCalculate<Integer>{

    IntegerNANDGate g1;
    IntegerNANDGate g2;
    IntegerNANDGate g3;
    IntegerNANDGate g4;

    public IntegerXORGate() {
        g1 = new IntegerNANDGate();
        g2 = new IntegerNANDGate();
        g3 = new IntegerNANDGate();
        g4 = new IntegerNANDGate();
    }

    @Override
    public Integer calculate(List<Integer> x) {
        assert x.size() == 2;
        Integer a = x.get(0);
        Integer b = x.get(1);

        Integer rg1 = g1.calculate(x);
        Integer rg2 = g1.calculate(Arrays.asList(a,rg1));
        Integer rg3 = g1.calculate(Arrays.asList(b,rg1));
        Integer rg4 = g1.calculate(Arrays.asList(rg2,rg3));

        return rg4;

    }

}
