package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
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
        "@org_tensorflow//tensorflow/core/dragnn/components/util:bulk_feature_extractor",
        "@org_tensorflow//tensorflow/core/dragnn:input_batch_cache",
        "@org_tensorflow//tensorflow/core/dragnn/protos:spec_proto",
        "@org_tensorflow//tensorflow/core/dragnn/protos:trace_proto",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
        "@org_tensorflow//tensorflow/core/syntaxnet:registry",
    ],
)

cc_library(
    name = "input_batch",
    hdrs = ["input_batch.h"],
    deps = [
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
    ],
)

cc_library(
    name = "transition_state",
    hdrs = ["transition_state.h"],
    deps = [
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
    ],
)
