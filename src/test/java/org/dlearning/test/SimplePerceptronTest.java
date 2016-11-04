package org.dlearning.test;

import lombok.extern.slf4j.Slf4j;
import org.dlearning.IntegerNANDGate;
import org.dlearning.IntegerXORGate;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

/**
 * Playground
 *
 * @author : Joao Costa (joaocarlosfilho@gmail.com) on 04/11/2016.
 */
@Slf4j
public class SimplePerceptronTest {
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

        Assert.assertEquals(1,ngr1.intValue());
        Assert.assertEquals(1,ngr2.intValue());
        Assert.assertEquals(1,ngr3.intValue());
        Assert.assertEquals(0,ngr4.intValue());

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

        Assert.assertEquals(0,xgr1.intValue());
        Assert.assertEquals(1,xgr2.intValue());
        Assert.assertEquals(1,xgr3.intValue());
        Assert.assertEquals(0,xgr4.intValue());

    }

}
