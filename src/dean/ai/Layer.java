package dean.ai;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Layer {

    public List<Neuron> neurons = new ArrayList<>();

    public Layer(int inputAmounts, int neuronAmount) {
        Random random = new Random();
        for (int i = 0; i < neuronAmount; i++) {
            List<Float> weights = new ArrayList<>();
            for (int j = 0; j < inputAmounts; j++) {
                weights.add(random.nextFloat() * 2f - 1f);
            }
            neurons.add(new Neuron(weights));
        }
    }

    public Layer(List<Float> inputs) {
        for (Float input : inputs) {
            this.neurons.add(new Neuron(input));
        }
    }
}
