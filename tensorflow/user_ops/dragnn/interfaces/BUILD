package(
    default_visibility = ["//visibility:public"],
    #features = ["-layering_check"],
)

cc_library(
    name = "cloneable_transition_state",
    hdrs = ["cloneable_transition_state.h"],
    deps = [":transition_state"],
)

cc_library(
    name = "component",
    hdrs = ["component.h"],
    deps = [
        ":transition_state",
        "//tensorflow/user_ops/dragnn/components/util:bulk_feature_extractor",
        "//tensorflow/user_ops/dragnn:input_batch_cache",
        "//tensorflow/user_ops/dragnn/protos:spec_proto_cc",
        "//tensorflow/user_ops/dragnn/protos:trace_proto_cc",
        "//tensorflow/user_ops/syntaxnet:base",
        "//tensorflow/user_ops/syntaxnet:registry",
    ],
)

cc_library(
    name = "input_batch",
    hdrs = ["input_batch.h"],
    deps = [
        "//tensorflow/user_ops/syntaxnet:base",
    ],
)

cc_library(
    name = "transition_state",
    hdrs = ["transition_state.h"],
    deps = [
        "//tensorflow/user_ops/syntaxnet:base",
    ],
)
