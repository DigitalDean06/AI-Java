package dean.ai;

import java.util.ArrayList;
import java.util.List;

public class Neuron {

    public final List<Float> cachedWeights;
    public final List<Float> cacheWeights = new ArrayList<>();
    public float value;
    public float gradient;

    public Neuron(List<Float> cachedWeights) {
        this.cachedWeights = cachedWeights;
        this.value = 0f;
        this.gradient = 0f;
    }

    public Neuron(float value) {
        this.cachedWeights = null;
        this.value = value;
        this.gradient = -1f;
    }

    public void updateWeights() {
        if (this.cachedWeights != null) {
            this.cachedWeights.clear();
            this.cachedWeights.addAll(this.cacheWeights);
            this.cacheWeights.clear();
        }
    }
}
