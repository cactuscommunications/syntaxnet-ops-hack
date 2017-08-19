import javafx.util.Pair;
import org.junit.Test;
import org.tensorflow.*;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

class ParseyMcParsefaceWrapper extends TensorflowModelWrapper<Tensor> {

    public ParseyMcParsefaceWrapper(final String savedModelDir, final List<String> outputOpNames) {
        super(savedModelDir, outputOpNames);
    }

    @Override
    protected List<Tensor> runModelImpl(final List<Pair<String, Tensor>> inputs, final SavedModelBundle bundle) {

        System.out.println("Fetching session...");

        // Retrieve runner from session
        final Session.Runner runner = bundle.session().runner();

        // Feed all inputs to the graph
        for (final Pair<String, Tensor> input : inputs) {
            runner.feed(input.getKey(), input.getValue());
        }

        final List<Tensor> outputs = new ArrayList<>();

        // Fetch all graph outputs
        for (final String outputOpName : this.outputOpNames) {

            final Tensor output = runner.fetch(outputOpName).run().get(0);
            outputs.add(output);
        }

        return outputs;
    }
}

public class TensorFlowModelWrapperTest {

    public TensorFlowModelWrapperTest() {
    }

    @Test
    public void testSyntaxnet() throws Exception {

        System.out.println(TensorFlowModelWrapperTest.class.getName());

        System.out.println("Loading model....");

        try (final ParseyMcParsefaceWrapper rrnWrapper =
                     new ParseyMcParsefaceWrapper(
                             "/mnt/c/Users/marhl/syntax_net_with_tensors/models/syntaxnet/SAVED_MODEL/",
                             Arrays.asList("output"))) {

            System.out.println("Model loaded...");


            final String[] mockdata =
                    {
                            "I love grapes.",
                            "Do you even lift bro?",
                            "Does your mother known that your out?",
                            "If you want to remove specific punctuation from a string, it will probably be best to explicitly remove exactly what you want like."
                    }; //

            final StringBuffer stringBuffer = new StringBuffer();

            for (int counter = 0; counter < mockdata.length; counter++) {
                stringBuffer.append(mockdata[counter]);
                stringBuffer.append('\n');
            }

            //final String inputString = stringBuffer.toString();
            final String inputString =
                    readFile("/mnt/c/Users/marhl/jni_with_ops_hack/tensorflow/test_data/test_document.txt", StandardCharsets.UTF_8);


            final Tensor mock_input = Tensor.create(inputString.getBytes(StandardCharsets.UTF_8));
            //System.out.println("INPUT STRING IS:\n" + inputString);

            final List<Pair<String, Tensor>> inputs = Arrays.asList(
                    new Pair<>("input", mock_input)
            );

            final List<Tensor> outputs = rrnWrapper.runModel(inputs);

            for (int counter = 0; counter < outputs.size(); counter++) {

                final Tensor tensor = outputs.get(counter);
                final byte[] out_bytes = tensor.bytesValue();

                final String string = new String(out_bytes, StandardCharsets.UTF_8);
                //System.out.println(string);
            }
        }
    }

    private static String readFile(final String path, final Charset encoding)
            throws IOException {
        byte[] encoded = Files.readAllBytes(Paths.get(path));
        return new String(encoded, encoding);
    }
}