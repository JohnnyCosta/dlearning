package org.dlearning;

import java.util.List;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

/**
 * Simple Perceptron for playing around
 *
 * @author : Joao Costa (joaocarlosfilho@gmail.com) on 04/11/2016.
 */
public class GenericPerceptron<T> implements InputCalculate<T>  {

    //  Weight
    protected List<T> w;
    // Perceptron Function
    protected UnaryOperator<T> f;
    // Multiplication Function
    protected BinaryOperator<T> m;
    // Sum Function
    protected BinaryOperator<T> s;
    // Bias value
    protected T b;
    // Identity init
    protected T init;

    public GenericPerceptron(List<T> weights, T bias, T identity, UnaryOperator<T> function, BinaryOperator<T> mult, BinaryOperator<T> sum) {
        w = weights;
        f = function;
        m = mult;
        s = sum;
        b = bias;
        init = identity;
    }

    @Override
    public T calculate(List<T> x) {
        assert (w.size() > 0);
        assert (w.size() == x.size());

        T vecmulti = init;
        for (int i = 0; i < w.size(); i++) {
            vecmulti = s.apply(vecmulti, m.apply(x.get(i), w.get(i)));
        }

        return f.apply(s.apply(vecmulti, b));
    }


}
