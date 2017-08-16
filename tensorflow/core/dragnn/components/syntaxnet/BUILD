package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
)

cc_library(
    name = "syntaxnet_component",
    srcs = ["syntaxnet_component.cc"],
    hdrs = ["syntaxnet_component.h"],
    deps = [
        ":syntaxnet_link_feature_extractor",
        ":syntaxnet_transition_state",
        "//tensorflow/core/dragnn/components/util:bulk_feature_extractor",
        "//tensorflow/core/dragnn:beam",
        "//tensorflow/core/dragnn:component_registry",
        "//tensorflow/core/dragnn:input_batch_cache",
        "//tensorflow/core/dragnn/interfaces:component",
        "//tensorflow/core/dragnn/interfaces:transition_state",
        "//tensorflow/core/dragnn/io:sentence_input_batch",
        "//tensorflow/core/dragnn/io:syntaxnet_sentence",
        "//tensorflow/core/dragnn/protos:data_proto",
        "//tensorflow/core/dragnn/protos:spec_proto",
        "//tensorflow/core/dragnn/protos:trace_proto",
        "//tensorflow/core/syntaxnet:base",
        "//tensorflow/core/syntaxnet:parser_transitions",
        "//tensorflow/core/syntaxnet:registry",
        "//tensorflow/core/syntaxnet:sparse_proto",
        "//tensorflow/core/syntaxnet:task_context",
        "//tensorflow/core/syntaxnet:task_spec_proto",
        "//tensorflow/core/syntaxnet:utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "syntaxnet_link_feature_extractor",
    srcs = ["syntaxnet_link_feature_extractor.cc"],
    hdrs = ["syntaxnet_link_feature_extractor.h"],
    deps = [
        "//tensorflow/core/dragnn/protos:spec_proto",
        "//tensorflow/core/syntaxnet:base",
        "//tensorflow/core/syntaxnet:embedding_feature_extractor",
        "//tensorflow/core/syntaxnet:parser_transitions",
        "//tensorflow/core/syntaxnet:task_context",
    ],
)

cc_library(
    name = "syntaxnet_transition_state",
    srcs = ["syntaxnet_transition_state.cc"],
    hdrs = ["syntaxnet_transition_state.h"],
    deps = [
        "//tensorflow/core/dragnn/interfaces:cloneable_transition_state",
        "//tensorflow/core/dragnn/interfaces:transition_state",
        "//tensorflow/core/dragnn/io:syntaxnet_sentence",
        "//tensorflow/core/dragnn/protos:trace_proto",
        "//tensorflow/core/syntaxnet:base",
        "//tensorflow/core/syntaxnet:parser_transitions",
    ],
)

## Test data.
#filegroup(
#    name = "testdata",
#    data = glob(["testdata/**"]),
#)
#
## Tests.
#cc_test(
#    name = "syntaxnet_component_test",
#    srcs = ["syntaxnet_component_test.cc"],
#    data = [":testdata"],
#    deps = [
#        ":syntaxnet_component",
#        "//dragnn/core:input_batch_cache",
#        "//dragnn/core/test:generic",
#        "//dragnn/core/test:mock_transition_state",
#        "//dragnn/io:sentence_input_batch",
#        "//syntaxnet:base",
#        "//syntaxnet:sentence_proto",
#        "//syntaxnet:test_main",
#    ],
#)
#
#cc_test(
#    name = "syntaxnet_link_feature_extractor_test",
#    srcs = ["syntaxnet_link_feature_extractor_test.cc"],
#    deps = [
#        ":syntaxnet_link_feature_extractor",
#        "//dragnn/core/test:generic",
#        "//dragnn/protos:spec_proto",
#        "//syntaxnet:task_context",
#        "//syntaxnet:test_main",
#    ],
#)
#
#cc_test(
#    name = "syntaxnet_transition_state_test",
#    srcs = ["syntaxnet_transition_state_test.cc"],
#    data = [":testdata"],
#    deps = [
#        ":syntaxnet_component",
#        ":syntaxnet_transition_state",
#        "//dragnn/core:input_batch_cache",
#        "//dragnn/core/test:generic",
#        "//dragnn/core/test:mock_transition_state",
#        "//dragnn/io:sentence_input_batch",
#        "//dragnn/protos:spec_proto",
#        "//syntaxnet:base",
#        "//syntaxnet:sentence_proto",
#        "//syntaxnet:test_main",
#    ],
#)
