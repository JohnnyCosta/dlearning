package org.dlearning.test;

import lombok.extern.slf4j.Slf4j;
import org.dlearning.IntegerNANDGate;
import org.dlearning.IntegerXORGate;
import org.dlearning.MultiLayer;
import org.dlearning.SigmoidNeuron;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Playground
 *
 * @author : Joao Costa (joaocarlosfilho@gmail.com) on 04/11/2016.
 */
@Slf4j
public class BinaryGatesTest {
    @Test
    public void testAPerceptron() {

        List<Integer> x1 = Arrays.asList(0, 0);
        List<Integer> x2 = Arrays.asList(0, 1);
        List<Integer> x3 = Arrays.asList(1, 0);
        List<Integer> x4 = Arrays.asList(1, 1);

        // NAND GATE
        IntegerNANDGate ngate = new IntegerNANDGate();
        Integer ngr1 = ngate.calculate(x1);
        Integer ngr2 = ngate.calculate(x2);
        Integer ngr3 = ngate.calculate(x3);
        Integer ngr4 = ngate.calculate(x4);

        log.info("NAND GATE");
        log.info("For input 1 '{}' produces '{}'", x1, ngr1);
        log.info("For input 2 '{}' produces '{}'", x2, ngr2);
        log.info("For input 3 '{}' produces '{}'", x3, ngr3);
        log.info("For input 4 '{}' produces '{}'", x4, ngr4);

        Assert.assertEquals(1, ngr1.intValue());
        Assert.assertEquals(1, ngr2.intValue());
        Assert.assertEquals(1, ngr3.intValue());
        Assert.assertEquals(0, ngr4.intValue());

        // XOR GATE
        IntegerXORGate xgate = new IntegerXORGate();
        Integer xgr1 = xgate.calculate(x1);
        Integer xgr2 = xgate.calculate(x2);
        Integer xgr3 = xgate.calculate(x3);
        Integer xgr4 = xgate.calculate(x4);

        log.info("XOR GATE");
        log.info("For input 1 '{}' produces '{}'", x1, xgr1);
        log.info("For input 2 '{}' produces '{}'", x2, xgr2);
        log.info("For input 3 '{}' produces '{}'", x3, xgr3);
        log.info("For input 4 '{}' produces '{}'", x4, xgr4);

        Assert.assertEquals(0, xgr1.intValue());
        Assert.assertEquals(1, xgr2.intValue());
        Assert.assertEquals(1, xgr3.intValue());
        Assert.assertEquals(0, xgr4.intValue());

        // Sigmoid neuron
        List<Double> sx1 = Arrays.asList(0d, 0d);
        List<Double> sx2 = Arrays.asList(0d, 1d);
        List<Double> sx3 = Arrays.asList(1d, 0d);
        List<Double> sx4 = Arrays.asList(1d, 1d);

        SigmoidNeuron sneuron = new SigmoidNeuron(Arrays.asList(-2d, -2d), 3d);
        Double sr1 = sneuron.calculate(sx1);
        Double sr2 = sneuron.calculate(sx2);
        Double sr3 = sneuron.calculate(sx3);
        Double sr4 = sneuron.calculate(sx4);

        log.info("Sigmoid");
        log.info("For input 1 '{}' produces '{}'", sx1, sr1);
        log.info("For input 2 '{}' produces '{}'", sx2, sr2);
        log.info("For input 3 '{}' produces '{}'", sx3, sr3);
        log.info("For input 4 '{}' produces '{}'", sx4, sr4);

        // Multilayer
        List<List<List<Double>>> weights = new ArrayList<>();
        List<List<Double>> layer1 = new ArrayList<>();
        layer1.add(Arrays.asList(0.1, 0.8));
        layer1.add(Arrays.asList(0.4, 0.6));
        weights.add(layer1);
        List<List<Double>> layer2 = new ArrayList<>();
        layer2.add(Arrays.asList(0.3, 0.9));
        weights.add(layer2);
        List<List<Double>> bias = new ArrayList<>();
        bias.add(Arrays.asList(0d, 0d));
        bias.add(Arrays.asList(0d));

        List layers = Arrays.asList(2, 2, 1);

        MultiLayer mlp = new MultiLayer(layers, weights, bias);
        List<Double> mx1 = Arrays.asList(0.35, 0.9);
        List<Double> out1 = mlp.calculate(mx1);

        log.info("Multilayer perceptron with size '{}'", layers);
        for (int i = 0; i < weights.size(); i++) {
            log.info("Layer '{}' weights '{}'", i, weights.get(i));
        }
        log.info("Bias '{}'", bias);
        log.info("For input 1 '{}' produces '{}'", mx1, out1);

    }

}
