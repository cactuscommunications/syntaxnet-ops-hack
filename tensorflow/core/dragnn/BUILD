package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
)

# Test data.
filegroup(
    name = "testdata",
    data = glob(["testdata/**"]),
)

cc_library(
    name = "beam",
    hdrs = ["beam.h"],
    deps = [
        "//tensorflow/core/dragnn/interfaces:cloneable_transition_state",
        "//tensorflow/core/dragnn/interfaces:transition_state",
        "//tensorflow/core/syntaxnet:base",
    ],
)

cc_library(
    name = "component_registry",
    srcs = ["component_registry.cc"],
    hdrs = ["component_registry.h"],
    deps = [
        "//tensorflow/core/dragnn/components/util:bulk_feature_extractor",
        "//tensorflow/core/dragnn/interfaces:component",
        "//tensorflow/core/dragnn/interfaces:input_batch",
        "//tensorflow/core/syntaxnet:registry",
        ":input_batch_cache",
        "//tensorflow/core/dragnn/protos:data_proto",
        "//tensorflow/core/dragnn/protos:spec_proto",
        "//tensorflow/core/dragnn/protos:trace_proto",
    ],
)

cc_library(
    name = "compute_session",
    hdrs = ["compute_session.h"],
    deps = [
        "//tensorflow/core/dragnn/components/util:bulk_feature_extractor",
        ":index_translator",
        "//tensorflow/core/dragnn/interfaces:component",
        "//tensorflow/core/dragnn/protos:data_proto",
        "//tensorflow/core/dragnn/protos:spec_proto",
        "//tensorflow/core/dragnn/protos:trace_proto",
    ],
)

cc_library(
    name = "compute_session_impl",
    srcs = ["compute_session_impl.cc"],
    hdrs = ["compute_session_impl.h"],
    deps = [
        ":compute_session",
        ":index_translator",
        ":input_batch_cache",
        "//tensorflow/core/dragnn/components/util:bulk_feature_extractor",
        "//tensorflow/core/dragnn/protos:data_proto",
        "//tensorflow/core/dragnn/protos:spec_proto",
        "//tensorflow/core/dragnn/protos:trace_proto",
        "//tensorflow/core/syntaxnet:base",
        "//tensorflow/core/syntaxnet:registry",
        "//util/utf8:unicodetext"
    ],
)

cc_library(
    name = "compute_session_pool",
    srcs = ["compute_session_pool.cc"],
    hdrs = ["compute_session_pool.h"],
    deps = [
        ":component_registry",
        ":compute_session",
        ":compute_session_impl",
        "//tensorflow/core/dragnn/protos:spec_proto",
        "//tensorflow/core/syntaxnet:base",
    ],
)

cc_library(
    name = "index_translator",
    srcs = ["index_translator.cc"],
    hdrs = ["index_translator.h"],
    deps = [
        "//tensorflow/core/syntaxnet:base",
        "//tensorflow/core/syntaxnet:registry",
        "//tensorflow/core/dragnn/interfaces:component",
        "//tensorflow/core/dragnn/interfaces:input_batch",
        "//tensorflow/core/dragnn/interfaces:transition_state",
        "//tensorflow/core/dragnn:input_batch_cache",
        "//tensorflow/core/dragnn/protos:data_proto",
        "//tensorflow/core/dragnn/protos:spec_proto",
        "//tensorflow/core/dragnn/protos:trace_proto",
        "//tensorflow/core/dragnn/components/util:bulk_feature_extractor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "input_batch_cache",
    hdrs = ["input_batch_cache.h"],
    deps = [
        "//tensorflow/core/dragnn/interfaces:input_batch",
        "//tensorflow/core/syntaxnet:base",
    ],
)

cc_library(
    name = "resource_container",
    hdrs = ["resource_container.h"],
    deps = ["//tensorflow/core/syntaxnet:base"],
)

# Tensorflow op kernel BUILD rules.

load(
    "//tensorflow/core/dragnn:tensorflow_ops.bzl",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)



#cc_library(
#    name = "compute_session_op",
#    srcs = [
#        "ops/compute_session_op.cc",
#    ],
#    hdrs = ["ops/compute_session_op.h"],
#    deps = [
#        ":compute_session",
#        ":resource_container",
#        "//tensorflow/core/syntaxnet:base",
#        "@org_tensorflow//third_party/eigen3",
#    ],
#)

#cc_library(
#    name = "dragnn_ops_cc",
#    srcs = [
#        "ops/dragnn_op_kernels.cc",
#        "ops/dragnn_ops.cc",
#    ],
#    deps = [
#        ":compute_session",
#        ":compute_session_op",
#        ":compute_session_pool",
#        ":resource_container",
#        "//tensorflow/core/dragnn/protos:data_proto",
#        "//tensorflow/core/dragnn/protos:spec_proto",
#        "//tensorflow/core/syntaxnet:base",
#        "@org_tensorflow//third_party/eigen3",
#    ],
#    alwayslink = 1,
#)

#cc_library(
#    name = "dragnn_bulk_ops_cc",
#    srcs = [
#        "ops/dragnn_bulk_op_kernels.cc",
#        "ops/dragnn_bulk_ops.cc",
#    ],
#    deps = [
#        "//tensorflow/core:compute_session_op_op_lib",
#        ":resource_container",
#        "//tensorflow/core/syntaxnet:base",
#        "@org_tensorflow//third_party/eigen3",
#    ],
#)

# Tensorflow kernel libraries, for use with unit tests.

tf_kernel_library(
    name = "dragnn_op_kernels",
    srcs = [
        "ops/dragnn_op_kernels.cc",
        "ops/dragnn_ops.cc",
    ],
    hdrs = [
    ],
    deps = [
        ":compute_session",
        "//tensorflow/core:compute_session_op_op_lib",
        ":compute_session_pool",
        ":resource_container",
        "//tensorflow/core/dragnn/protos:spec_proto",
        "//tensorflow/core/syntaxnet:base",
        "@org_tensorflow//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "dragnn_bulk_op_kernels",
    srcs = [
        "ops/dragnn_bulk_op_kernels.cc",
        "ops/dragnn_bulk_ops.cc",
    ],
    hdrs = [
    ],
    deps = [
        ":compute_session",
        "//tensorflow/core:compute_session_op_op_lib",
        ":compute_session_pool",
        ":resource_container",
        "//tensorflow/core/dragnn/components/util:bulk_feature_extractor",
        "//tensorflow/core/dragnn/protos:spec_proto",
        "//tensorflow/core/syntaxnet:base",
        "//tensorflow/core:protos_all_cc",
        "@org_tensorflow//third_party/eigen3",
    ],
)
