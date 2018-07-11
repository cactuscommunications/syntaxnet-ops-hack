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
        "//tensorflow/user_ops/dragnn/components/util:bulk_feature_extractor",
        "//tensorflow/user_ops/dragnn:beam",
        "//tensorflow/user_ops/dragnn:component_registry",
        "//tensorflow/user_ops/dragnn:input_batch_cache",
        "//tensorflow/user_ops/dragnn/interfaces:component",
        "//tensorflow/user_ops/dragnn/interfaces:transition_state",
        "//tensorflow/user_ops/dragnn/io:sentence_input_batch",
        "//tensorflow/user_ops/dragnn/io:syntaxnet_sentence",
        "//tensorflow/user_ops/dragnn/protos:data_proto",
        "//tensorflow/user_ops/dragnn/protos:spec_proto",
        "//tensorflow/user_ops/dragnn/protos:trace_proto",
        "//tensorflow/user_ops/syntaxnet:base",
        "//tensorflow/user_ops/syntaxnet:parser_transitions",
        "//tensorflow/user_ops/syntaxnet:registry",
        "//tensorflow/user_ops/syntaxnet:sparse_proto",
        "//tensorflow/user_ops/syntaxnet:task_context",
        "//tensorflow/user_ops/syntaxnet:task_spec_proto",
        "//tensorflow/user_ops/syntaxnet:utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "syntaxnet_link_feature_extractor",
    srcs = ["syntaxnet_link_feature_extractor.cc"],
    hdrs = ["syntaxnet_link_feature_extractor.h"],
    deps = [
        "//tensorflow/user_ops/dragnn/protos:spec_proto",
        "//tensorflow/user_ops/syntaxnet:base",
        "//tensorflow/user_ops/syntaxnet:embedding_feature_extractor",
        "//tensorflow/user_ops/syntaxnet:parser_transitions",
        "//tensorflow/user_ops/syntaxnet:task_context",
    ],
)

cc_library(
    name = "syntaxnet_transition_state",
    srcs = ["syntaxnet_transition_state.cc"],
    hdrs = ["syntaxnet_transition_state.h"],
    deps = [
        "//tensorflow/user_ops/dragnn/interfaces:cloneable_transition_state",
        "//tensorflow/user_ops/dragnn/interfaces:transition_state",
        "//tensorflow/user_ops/dragnn/io:syntaxnet_sentence",
        "//tensorflow/user_ops/dragnn/protos:trace_proto",
        "//tensorflow/user_ops/syntaxnet:base",
        "//tensorflow/user_ops/syntaxnet:parser_transitions",
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
#        "//dragnn/user_ops:input_batch_cache",
#        "//dragnn/user_ops/test:generic",
#        "//dragnn/user_ops/test:mock_transition_state",
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
