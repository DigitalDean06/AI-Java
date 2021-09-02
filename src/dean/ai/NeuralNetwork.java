package dean.ai;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuralNetwork {

    public final List<Layer> layers = new ArrayList<>();
    public final List<TrainingData> trainingDataList = new ArrayList<>();

    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork();
        network.createLayers();
        network.createTrainingData();
        System.out.println("=".repeat(18));
        System.out.println("Output before training");
        System.out.println("=".repeat(18));
        for (TrainingData data : network.trainingDataList) {
            network.forward(data);
            System.out.println(network.layers.get(network.layers.size() - 1).neurons.get(0).value);
        }
        long currentTime = System.currentTimeMillis();
        int trainingCount = 1000000;
        float learningRate = 0.1f;
        network.train(trainingCount, learningRate);
        System.out.printf("Finished %s training with learning rate %s in %s seconds!\n",trainingCount, learningRate, ((System.currentTimeMillis() - currentTime) / 1000f));
        System.out.println("=".repeat(18));
        System.out.println("Output after training");
        System.out.println("=".repeat(18));
        for (TrainingData data : network.trainingDataList) {
            network.forward(data);
            System.out.println(network.layers.get(network.layers.size() - 1).neurons.get(0).value);
        }
    }

    public void createLayers() {
        this.layers.add(null); // input layer
        this.layers.add(new Layer(2, 6)); // hidden layer
        this.layers.add(new Layer(6, 1)); // output layer

    }

    public void createTrainingData() {
        this.trainingDataList.add(new TrainingData(new ArrayList<>(Arrays.asList(0f, 0f)), new ArrayList<>(List.of(1f))));
        this.trainingDataList.add(new TrainingData(new ArrayList<>(Arrays.asList(0f, 1f)), new ArrayList<>(List.of(0f))));
        this.trainingDataList.add(new TrainingData(new ArrayList<>(Arrays.asList(1f, 0f)), new ArrayList<>(List.of(0f))));
        this.trainingDataList.add(new TrainingData(new ArrayList<>(Arrays.asList(1f, 1f)), new ArrayList<>(List.of(1f))));
    }

    public void forward(TrainingData data) {
        this.layers.set(0, new Layer(data.data()));
        for (int i = 1; i < this.layers.size(); i++) {
            for (int j = 0; j < layers.get(i).neurons.size(); j++) {
                float sum = 0f;
                for (int k = 0; k < layers.get(i - 1).neurons.size(); k++) {
                    sum += layers.get(i - 1).neurons.get(k).value * layers.get(i).neurons.get(j).cachedWeights.get(k);
                }
                layers.get(i).neurons.get(j).value = (float) (Math.tanh(sum) + 1f) / 2f;
            }
        }

    }

    public void backward(TrainingData data, float learningRate) {
        Layer outputLayer = this.layers.get(this.layers.size() - 1);
        for (int i = 0; i < outputLayer.neurons.size(); i++) {
            Neuron neuron = outputLayer.neurons.get(i);
            float output = neuron.value;
            float target = data.expectedOutputs().get(i);
            float derivative = output - target;
            float delta = derivative * (output * (1f - output));
            neuron.gradient = delta;
            if (neuron.cachedWeights != null) {
                for (int j = 0; j < neuron.cachedWeights.size(); j++) {
                    float previousOutput = this.layers.get(this.layers.size() - 2).neurons.get(j).value;
                    float error = delta * previousOutput;
                    neuron.cacheWeights.add(j, neuron.cachedWeights.get(j) - error * learningRate);
                }
            }
        }
        for (int i = this.layers.size() - 2; i > 0; i--) {
            Layer layer = this.layers.get(i);
            for (int j = 0; j < layer.neurons.size(); j++) {
                Neuron neuron = layer.neurons.get(j);
                float output = neuron.value;
                float derivative = sumGradient(j, i + 1);
                float delta = derivative * (output * (1f - output));
                neuron.gradient = delta;
                if (neuron.cachedWeights != null) {
                    for (int k = 0; k < neuron.cachedWeights.size(); k++) {
                        float previousOutput = this.layers.get(i - 1).neurons.get(k).value;
                        float error = delta * previousOutput;
                        neuron.cacheWeights.add(k, neuron.cachedWeights.get(k) - error * learningRate);
                    }
                }
            }
        }
        for (Layer layer : this.layers) {
            for (Neuron neuron : layer.neurons) {
                neuron.updateWeights();
            }
        }
    }

    public float sumGradient(int neuronIndex, int layerIndex) {
        float gradientSum = 0f;
        Layer layer = this.layers.get(layerIndex);
        for (int i = 0; i < layer.neurons.size(); i++) {
            Neuron neuron = layer.neurons.get(i);
            if (neuron.cachedWeights != null) {
                gradientSum += neuron.cachedWeights.get(neuronIndex) * neuron.gradient;
            }
        }
        return gradientSum;
    }

    public void train(int count, float learningRate) {
        for (int i = 0; i < count; i++) {
            for (TrainingData data : this.trainingDataList) {
                this.forward(data);
                this.backward(data, learningRate);
            }
        }
    }
}
