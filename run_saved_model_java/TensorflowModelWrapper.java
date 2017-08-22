import org.tensorflow.*;

import java.util.ArrayList;
import java.util.List;

public abstract class TensorflowModelWrapper<IN, OUT> implements AutoCloseable {

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

    protected abstract OUT runModelImpl(final IN inputs, final Session session);

    public final OUT runModel(final IN input) {
        return runModelImpl(input, bundle.session());
    }

    @Override
    public void close() {
        if (bundle != null) {
            bundle.close();
        }
    }

    protected static class TensorflowModelWrapperException extends IllegalArgumentException {
        public TensorflowModelWrapperException(final String message, final Throwable cause) {
            super(message, cause);
        }
    }

}