package org.dlearning;

import org.apache.commons.math3.util.FastMath;

import java.util.List;

/**
 * Softmax function
 *
 * @author : Joao Costa (joaocarlosfilho@gmail.com) on 04/11/2016.
 */
public class SoftmaxFunction {

    private static Double softmax(List<Double> z, Integer j) {
        Double a = FastMath.exp(z.get(j));

        Double temp = 0d;
        for (int k = 0; j < z.size(); k++) {
            temp += FastMath.exp(z.get(k));
        }

        return a / temp;
    }

    public static Double calculate(List<Double> z, Integer j) {
        return softmax(z, j);
    }
}
