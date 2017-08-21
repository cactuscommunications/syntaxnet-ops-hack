import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;

public class ParseyMcParsefaceWrapper extends TensorflowModelWrapper<String, String> {

    public ParseyMcParsefaceWrapper(final String savedModelDir) {
        super(savedModelDir, Arrays.asList("output"));
    }

    @Override
    protected String runModelImpl(final String input, final SavedModelBundle bundle) {

        // User information...
        System.out.println("Fetching session...");

        // Retrieve runner from session
        final Session.Runner runner = bundle.session().runner();

        // Create input Tensor from String input
        final Tensor inputTensor = Tensor.create(input.getBytes(StandardCharsets.UTF_8));

        // Feed all inputs to the graph
        runner.feed("input", inputTensor);

        // Fetch all graph outputs
        final Tensor outputTensor = runner.fetch(this.outputOpNames.get(0)).run().get(0);

        // Fetch output bytes from tensor:
        final byte[] outputBytes = outputTensor.bytesValue();

        // Decode bytes to String
        final String output = new String(outputBytes, StandardCharsets.UTF_8);

        return output;
    }
}