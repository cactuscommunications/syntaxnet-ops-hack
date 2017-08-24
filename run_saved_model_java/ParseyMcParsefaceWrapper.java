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

        final int ONE_MB = 1024 * 128;
        final int CHUNK_SIZE = ONE_MB;
        final Splitter splitter = Splitter.on('\n').trimResults().omitEmptyStrings();

        // User information...
        System.out.println("Fetching session... Input is: " + input.length());

        // Retrieve runner from session
        final StringBuilder inputBuilder = new StringBuilder();
        final StringBuilder outputBuilder = new StringBuilder();

        // Build chunks of CHUNK_SIZE and feed them to parsey
        for (String substring : splitter.split(input)) {

            if (inputBuilder.length() + substring.length() >= CHUNK_SIZE - 1) {

                final String output = runParseyMcParseface(inputBuilder.toString(), session);
                outputBuilder.append(output);
                inputBuilder.setLength(0);
            }

            inputBuilder.append(substring);
            inputBuilder.append('\n');
        }

        // Do a final run on the remaining text
        if(inputBuilder.length() > 0)
        {
            final String output = runParseyMcParseface(inputBuilder.toString(), session);
            outputBuilder.append(output);
        }

        return outputBuilder.toString();
    }

    private String runParseyMcParseface(final String input, final Session session)
    {
        System.out.println("About to run on " + input.length() + " chars");

        final Session.Runner runner = session.runner();

        // Create input Tensor from String input
        final Tensor inputTensor = Tensor.create(input.getBytes(StandardCharsets.UTF_8));

        // Feed all inputs to the graph
        runner.feed("input", inputTensor);

        // Fetch all graph outputs
        final Tensor outputTensor = runner.fetch(this.outputOpNames.get(0)).run().get(0);

        // Fetch output bytes from tensor:
        final byte[] outputBytes = outputTensor.bytesValue();

        // Close tensors (i.e. free native memory)
        outputTensor.close();
        inputTensor.close();

        // Decode bytes to String
        return new String(outputBytes, StandardCharsets.UTF_8);
    }
}