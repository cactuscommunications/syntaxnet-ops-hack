import com.google.common.base.Splitter;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;


public class ParseyMcParsefaceWrapper extends TensorflowModelWrapper<String, String> {

    public ParseyMcParsefaceWrapper(final String savedModelDir) {
        super(savedModelDir, Arrays.asList("output"));
    }

    /**
     * @param input   a string of newline separated sentences
     * @param session a saved model bundle
     * @return
     */
    @Override
    protected String runModelImpl(final String input, final Session session) {

        final int ONE_MB = 1024 * 1024;
        final Splitter splitter = Splitter.fixedLength(ONE_MB);

        // User information...
        System.out.println("Fetching session...");

        // Retrieve runner from session
        final Session.Runner runner = session.runner();

        final StringBuilder outputBuilder = new StringBuilder();

        for (String substring : splitter.splitToList(input)) {

            // Create input Tensor from String input
            final Tensor inputTensor = Tensor.create(substring.getBytes(StandardCharsets.UTF_8));

            // Feed all inputs to the graph
            runner.feed("input", inputTensor);

            // Fetch all graph outputs
            final Tensor outputTensor = runner.fetch(this.outputOpNames.get(0)).run().get(0);

            // Fetch output bytes from tensor:
            final byte[] outputBytes = outputTensor.bytesValue();

            // Decode bytes to String
            final String output = new String(outputBytes, StandardCharsets.UTF_8);

            outputBuilder.append(output);
        }

        return outputBuilder.toString();
    }
}