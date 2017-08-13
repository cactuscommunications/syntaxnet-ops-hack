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
        "@org_tensorflow//tensorflow/core/dragnn/interfaces:cloneable_transition_state",
        "@org_tensorflow//tensorflow/core/dragnn/interfaces:transition_state",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
    ],
)

cc_library(
    name = "component_registry",
    srcs = ["component_registry.cc"],
    hdrs = ["component_registry.h"],
    deps = [
        "@org_tensorflow//tensorflow/core/dragnn/interfaces:component",
        "@org_tensorflow//tensorflow/core/syntaxnet:registry",
    ],
)

cc_library(
    name = "compute_session",
    hdrs = ["compute_session.h"],
    deps = [
        "@org_tensorflow//tensorflow/core/dragnn/components/util:bulk_feature_extractor",
        "@org_tensorflow//tensorflow/core/dragnn:index_translator",
        "@org_tensorflow//tensorflow/core/dragnn/interfaces:component",
        "@org_tensorflow//tensorflow/core/dragnn/protos:spec_proto",
        "@org_tensorflow//tensorflow/core/dragnn/protos:trace_proto",
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
        "@org_tensorflow//tensorflow/core/dragnn/components/util:bulk_feature_extractor",
        "@org_tensorflow//tensorflow/core/dragnn/protos:data_proto",
        "@org_tensorflow//tensorflow/core/dragnn/protos:spec_proto",
        "@org_tensorflow//tensorflow/core/dragnn/protos:trace_proto",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
        "@org_tensorflow//tensorflow/core/syntaxnet:registry",
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
        "@org_tensorflow//tensorflow/core/dragnn/protos:spec_proto",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
    ],
)

cc_library(
    name = "index_translator",
    srcs = ["index_translator.cc"],
    hdrs = ["index_translator.h"],
    deps = [
        "@org_tensorflow//tensorflow/core/dragnn/interfaces:component",
        "@org_tensorflow//tensorflow/core/dragnn/interfaces:transition_state",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
    ],
)

cc_library(
    name = "input_batch_cache",
    hdrs = ["input_batch_cache.h"],
    deps = [
        "@org_tensorflow//tensorflow/core/dragnn/interfaces:input_batch",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
    ],
)

cc_library(
    name = "resource_container",
    hdrs = ["resource_container.h"],
    deps = ["@org_tensorflow//tensorflow/core/syntaxnet:base"],
)

# Tensorflow op kernel BUILD rules.

load(
    "//tensorflow/core/dragnn:tensorflow_ops.bzl",
    "tf_gen_op_libs",
    #"tf_gen_op_wrapper_py",
    "tf_kernel_library",
)



cc_library(
    name = "compute_session_op",
    srcs = [
        "ops/compute_session_op.cc",
    ],
    hdrs = ["ops/compute_session_op.h"],
    deps = [
        ":compute_session",
        ":resource_container",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
        "@org_tensorflow//third_party/eigen3",
    ],
)

cc_library(
    name = "dragnn_ops_cc",
    srcs = [
        "ops/dragnn_op_kernels.cc",
        "ops/dragnn_ops.cc",
    ],
    deps = [
        ":compute_session",
        ":compute_session_op",
        ":compute_session_pool",
        ":resource_container",
        "@org_tensorflow//tensorflow/core/dragnn/protos:data_proto",
        "@org_tensorflow//tensorflow/core/dragnn/protos:spec_proto",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
        "@org_tensorflow//third_party/eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dragnn_bulk_ops_cc",
    srcs = [
        "ops/dragnn_bulk_op_kernels.cc",
        "ops/dragnn_bulk_ops.cc",
    ],
    deps = [
        ":compute_session_op",
        ":resource_container",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
        "@org_tensorflow//third_party/eigen3",
    ],
)

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
        ":compute_session_op",
        ":compute_session_pool",
        ":resource_container",
        "@org_tensorflow//tensorflow/core/dragnn/protos:data_proto",
        "@org_tensorflow//tensorflow/core/dragnn/protos:spec_proto",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
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
        ":compute_session_op",
        ":compute_session_pool",
        ":resource_container",
        "@org_tensorflow//tensorflow/core/dragnn/components/util:bulk_feature_extractor",
        "@org_tensorflow//tensorflow/core/dragnn/protos:spec_proto",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//third_party/eigen3",
    ],
)
