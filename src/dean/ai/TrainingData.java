package dean.ai;

import java.util.List;

public record TrainingData(List<Float> data, List<Float> expectedOutputs) {

}
