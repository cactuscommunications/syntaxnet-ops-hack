import javafx.util.Pair;
import org.tensorflow.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public abstract class TensorflowModelWrapper<InputType> implements AutoCloseable {

    private final SavedModelBundle bundle;
    protected final List<String> outputOpNames;

    public TensorflowModelWrapper(final String savedModelDir, final List<String> outputOpNames) {

        this.bundle = SavedModelBundle.load(savedModelDir, "serve");
        this.outputOpNames = outputOpNames == null ? new ArrayList<>() : outputOpNames;
    }

    public final List<String> getOutputOpNames() {
        return this.outputOpNames;
    }

    protected final Graph getGraph() {
        return bundle.graph();
    }

    protected abstract List<Tensor> runModelImpl(final List<Pair<String, Tensor>> inputs, final SavedModelBundle bundle);

    public final List<Tensor> runModel(final List<Pair<String, Tensor>> inputs) {
        return runModelImpl(inputs, bundle);
    }

    @Override
    public void close() {
        if (bundle != null) {
            bundle.close();
        }
    }

    public static class TensorflowModelWrapperException extends IllegalArgumentException {
        public TensorflowModelWrapperException(final String message, final Throwable cause) {
            super(message, cause);
        }
    }

}