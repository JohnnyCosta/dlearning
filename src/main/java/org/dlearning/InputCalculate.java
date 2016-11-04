package org.dlearning;

import java.util.List;

/**
 * Inteface for all classes that calculates from input
 *
 * @author : Joao Costa (joaocarlosfilho@gmail.com) on 04/11/2016.
 */
public interface InputCalculate<T> {

    public T calculate(List<T> x);
}
