package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
)

cc_library(
    name = "stateless_component",
    srcs = ["stateless_component.cc"],
    deps = [
        "//tensorflow/core/dragnn:component_registry",
        "//tensorflow/core/dragnn/interfaces:component",
        "//tensorflow/core/dragnn/interfaces:transition_state",
        "//tensorflow/core/dragnn/io:sentence_input_batch",
        "//tensorflow/core/dragnn/protos:data_proto",
        "//tensorflow/core/syntaxnet:base",
    ],
    alwayslink = 1,
)

#cc_test(
#    name = "stateless_component_test",
#    srcs = ["stateless_component_test.cc"],
#    deps = [
#        ":stateless_component",
#        "//tensorflow/core/dragnn:component_registry",
#        "//tensorflow/core/dragnn:input_batch_cache",
#        "//tensorflow/core/dragnn/test:generic",
#        "//tensorflow/core/dragnn/test:mock_transition_state",
#        "//tensorflow/core/dragnn/io:sentence_input_batch",
#        "//tensorflow/core/syntaxnet:base",
#        "//tensorflow/core/syntaxnet:sentence_proto",
#        "//tensorflow/core/syntaxnet:test_main",
#    ],
#)
