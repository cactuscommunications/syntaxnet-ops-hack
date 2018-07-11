package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
)

cc_library(
    name = "stateless_component",
    srcs = ["stateless_component.cc"],
    deps = [
        "//tensorflow/user_ops/dragnn:component_registry",
        "//tensorflow/user_ops/dragnn/interfaces:component",
        "//tensorflow/user_ops/dragnn/interfaces:transition_state",
        "//tensorflow/user_ops/dragnn/io:sentence_input_batch",
        "//tensorflow/user_ops/dragnn/protos:data_proto",
        "//tensorflow/user_ops/syntaxnet:base",
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
