package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
)

cc_library(
    name = "stateless_component",
    srcs = ["stateless_component.cc"],
    deps = [
        "@org_tensorflow//tensorflow/core/dragnn:component_registry",
        "@org_tensorflow//tensorflow/core/dragnn/interfaces:component",
        "@org_tensorflow//tensorflow/core/dragnn/interfaces:transition_state",
        "@org_tensorflow//tensorflow/core/dragnn/io:sentence_input_batch",
        "@org_tensorflow//tensorflow/core/dragnn/protos:data_proto",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
    ],
    alwayslink = 1,
)

#cc_test(
#    name = "stateless_component_test",
#    srcs = ["stateless_component_test.cc"],
#    deps = [
#        ":stateless_component",
#        "@org_tensorflow//tensorflow/core/dragnn:component_registry",
#        "@org_tensorflow//tensorflow/core/dragnn:input_batch_cache",
#        "@org_tensorflow//tensorflow/core/dragnn/test:generic",
#        "@org_tensorflow//tensorflow/core/dragnn/test:mock_transition_state",
#        "@org_tensorflow//tensorflow/core/dragnn/io:sentence_input_batch",
#        "@org_tensorflow//tensorflow/core/syntaxnet:base",
#        "@org_tensorflow//tensorflow/core/syntaxnet:sentence_proto",
#        "@org_tensorflow//tensorflow/core/syntaxnet:test_main",
#    ],
#)
