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
        "@org_tensorflow//tensorflow/core/dragnn/components/util:bulk_feature_extractor",
        "@org_tensorflow//tensorflow/core/dragnn:beam",
        "@org_tensorflow//tensorflow/core/dragnn:component_registry",
        "@org_tensorflow//tensorflow/core/dragnn:input_batch_cache",
        "@org_tensorflow//tensorflow/core/dragnn/interfaces:component",
        "@org_tensorflow//tensorflow/core/dragnn/interfaces:transition_state",
        "@org_tensorflow//tensorflow/core/dragnn/io:sentence_input_batch",
        "@org_tensorflow//tensorflow/core/dragnn/io:syntaxnet_sentence",
        "@org_tensorflow//tensorflow/core/dragnn/protos:data_proto",
        "@org_tensorflow//tensorflow/core/dragnn/protos:spec_proto",
        "@org_tensorflow//tensorflow/core/dragnn/protos:trace_proto",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
        "@org_tensorflow//tensorflow/core/syntaxnet:parser_transitions",
        "@org_tensorflow//tensorflow/core/syntaxnet:registry",
        "@org_tensorflow//tensorflow/core/syntaxnet:sparse_proto",
        "@org_tensorflow//tensorflow/core/syntaxnet:task_context",
        "@org_tensorflow//tensorflow/core/syntaxnet:task_spec_proto",
        "@org_tensorflow//tensorflow/core/syntaxnet:utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "syntaxnet_link_feature_extractor",
    srcs = ["syntaxnet_link_feature_extractor.cc"],
    hdrs = ["syntaxnet_link_feature_extractor.h"],
    deps = [
        "@org_tensorflow//tensorflow/core/dragnn/protos:spec_proto",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
        "@org_tensorflow//tensorflow/core/syntaxnet:embedding_feature_extractor",
        "@org_tensorflow//tensorflow/core/syntaxnet:parser_transitions",
        "@org_tensorflow//tensorflow/core/syntaxnet:task_context",
    ],
)

cc_library(
    name = "syntaxnet_transition_state",
    srcs = ["syntaxnet_transition_state.cc"],
    hdrs = ["syntaxnet_transition_state.h"],
    deps = [
        "@org_tensorflow//tensorflow/core/dragnn/interfaces:cloneable_transition_state",
        "@org_tensorflow//tensorflow/core/dragnn/interfaces:transition_state",
        "@org_tensorflow//tensorflow/core/dragnn/io:syntaxnet_sentence",
        "@org_tensorflow//tensorflow/core/dragnn/protos:trace_proto",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
        "@org_tensorflow//tensorflow/core/syntaxnet:parser_transitions",
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
