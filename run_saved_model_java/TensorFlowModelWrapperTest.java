import javafx.util.Pair;
import org.junit.Test;
import org.tensorflow.*;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class TensorFlowModelWrapperTest {

    public TensorFlowModelWrapperTest() {
    }

    @Test
    public void testSyntaxnet() throws Exception {

        System.out.println("Loading model....");

        try (final ParseyMcParsefaceWrapper rrnWrapper =
                     new ParseyMcParsefaceWrapper(
                             "/mnt/c/Users/marhl/syntax_net_with_tensors/models/syntaxnet/SAVED_MODEL/")) {

            System.out.println("Model loaded...");

            /*final String[] mockdata =
                    {
                            "I love grapes.",
                            "Do you even lift bro?",
                            "Does your mother known that your out?",
                            "If you want to remove specific punctuation from a string, it will probably be best to explicitly remove exactly what you want like."
                    }; //*/

            /*final StringBuffer stringBuffer = new StringBuffer();

            for (int counter = 0; counter < mockdata.length; counter++) {
                stringBuffer.append(mockdata[counter]);
                stringBuffer.append('\n');
            }*/

            //final String inputString = stringBuffer.toString();
            final String inputString =
                    readFile("/mnt/c/Users/marhl/jni_with_ops_hack/tensorflow/test_data/test_document.txt", StandardCharsets.UTF_8);

            final String output = rrnWrapper.runModel(inputString);
            //System.out.println(output);
        }
    }

    private static String readFile(final String path, final Charset encoding)
            throws IOException {
        final byte[] encoded = Files.readAllBytes(Paths.get(path));
        return new String(encoded, encoding);
    }
}