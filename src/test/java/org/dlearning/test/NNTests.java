package org.dlearning.test;

import lombok.extern.slf4j.Slf4j;
import org.dlearning.NeuralNetwork;
import org.dlearning.Training;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Neural Network tests
 *
 * @author : Joao Costa (joaocarlosfilho@gmail.com) on 15/11/2016.
 */
@Slf4j
public class NNTests {

    @Test
    public void simpleNetwork() {
        log.info("Simple Neural network");
        List<List<List<Double>>> nnweights = new ArrayList<>();
        List<List<Double>> nnlayer1 = new ArrayList<>();
        nnlayer1.add(Arrays.asList(0.15, 0.20));
        nnlayer1.add(Arrays.asList(0.25, 0.30));
        nnweights.add(nnlayer1);
        List<List<Double>> nnlayer2 = new ArrayList<>();
        nnlayer2.add(Arrays.asList(0.40, 0.45));
        nnlayer2.add(Arrays.asList(0.50, 0.55));
        nnweights.add(nnlayer2);

        List bias = Arrays.asList(.35, .60);

        List nnlayers = Arrays.asList(2, 2, 2);

        NeuralNetwork nn = new NeuralNetwork(nnlayers, nnweights, bias, true);

        List<Training<Double>> trainings = new ArrayList<>();
        trainings.add(
                // Input
                new <Double>Training(Arrays.asList(0.05, 0.10),
                        // Expected output
                        Arrays.asList(0.01, 0.99)));


        nn.train(trainings, 0.5, 0.0001, 10000);

        List<List<Double>> nnout = nn.calculate(Arrays.asList(0.05, 0.10));

        log.info("Output  is '{}'", nnout.get(nnout.size()-1));
    }

    @Test
    public void trainSimplePattern() {
        log.info("Neural network with Simple Pattern");

        List nnlayers = Arrays.asList(2, 2, 1);

        NeuralNetwork nn = new NeuralNetwork(nnlayers, null, null, true);

        List<Training<Double>> trainings = new ArrayList<>();

        trainings.add(
                // Input
                new <Double>Training(Arrays.asList(0d, 1d),
                        // Expected output
                        Arrays.asList(0d))
        );


        trainings.add(
                // Input
                new <Double>Training(Arrays.asList(1d, 0d),
                        // Expected output
                        Arrays.asList(1d))
        );

        List<List<List<Double>>> currentWeights = nn.getCurrentWeights();
        for (List<List<Double>> lw : currentWeights) {
            log.info(lw.toString());
        }

        nn.trainBatch(trainings, 1, 0.01, 10000);

        currentWeights = nn.getCurrentWeights();
        for (List<List<Double>> lw : currentWeights) {
            log.info(lw.toString());
        }

        List<List<Double>> nnout = nn.calculate(Arrays.asList(0d, 1d));

        log.info("Output  is '{}'", nnout.get(nnout.size()-1));

        nnout = nn.calculate(Arrays.asList(1d, 0d));

        log.info("Output  is '{}'", nnout.get(nnout.size()-1));

    }

    @Test
    public void trainMoreComplexPattern() {
        log.info("Neural network with a more complex pattern");
        List<List<List<Double>>> nnweights = new ArrayList<>();
        List<List<Double>> nnlayer1 = new ArrayList<>();
        nnlayer1.add(Arrays.asList(1d, 1d, 1d, 1d));
        nnlayer1.add(Arrays.asList(1d, 1d, 1d, 1d));
        nnlayer1.add(Arrays.asList(1d, 1d, 1d, 1d));
        nnweights.add(nnlayer1);
        List<List<Double>> nnlayer2 = new ArrayList<>();
        nnlayer2.add(Arrays.asList(0.1, 0.1, 0.1));
        nnlayer2.add(Arrays.asList(0.1, 0.1, 0.1));
        nnweights.add(nnlayer2);

        List nnlayers = Arrays.asList(4, 4,4, 2);

        NeuralNetwork nn = new NeuralNetwork(nnlayers, null, null, true);

        List<Training<Double>> trainings = new ArrayList<>();
        trainings.add(
                // Input
                new <Double>Training(Arrays.asList(1d, 0d, 0d, 1d),
                        // Expected output
                        Arrays.asList(1d, 0d))
        );
        trainings.add(
                // Input
                new <Double>Training(Arrays.asList(0d, 1d, 1d, 0d),
                        // Expected output
                        Arrays.asList(0d, 1d))
        );

        nn.train(trainings, 1.5, 0.01, 100000);

        List<List<Double>> nnout = nn.calculate(Arrays.asList(1d, 0d, 0d, 1d));

        log.info("Output  is '{}'", nnout.get(nnout.size()-1));

        nnout = nn.calculate(Arrays.asList(0d, 1d, 1d, 0d));

        log.info("Output  is '{}'", nnout.get(nnout.size()-1));
    }
}
