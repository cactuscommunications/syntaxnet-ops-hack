# Description:
# An example for custom op and kernel defined as a TensorFlow plugin.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

# All syntaxnet/dragnn ops 
tf_custom_op_library(
    name = "syntaxnet_dragnn_ops.so",
    srcs = [
	"ops/compute_session_op.h",
    	"ops/dragnn_op_kernels.cc",
	"ops/dragnn_bulk_op_kernels.cc", 
    ],
    deps = [
	"//util/utf8:unicodetext",
        "//tensorflow/user_ops/syntaxnet:document_filters",
        "//tensorflow/user_ops/syntaxnet:lexicon_builder",
        "//tensorflow/user_ops/syntaxnet:reader_ops",
        "//tensorflow/user_ops/syntaxnet:unpack_sparse_features",
	"//tensorflow/user_ops/syntaxnet:registry",
        "//tensorflow/user_ops/dragnn/protos:data_proto_cc",
        "//tensorflow/user_ops/dragnn/protos:trace_proto_cc",
        "//tensorflow/user_ops/dragnn/protos:spec_proto_cc",
	"//tensorflow/user_ops/dragnn:compute_session",
        "//tensorflow/user_ops/dragnn:resource_container",
        "//tensorflow/user_ops/dragnn:compute_session_pool",
        "//tensorflow/user_ops/dragnn/interfaces:component",
        "//tensorflow/user_ops/dragnn:index_translator",
    ],
)

#tf_custom_op_library(
#    name = "syntaxnet_dragnn_ops.so",
#    srcs = [],
#    deps = [
#	":parser_ops",
#	":compute_session_op",
#	":dragnn_ops",
#	":dragnn_bulk_ops",
#    ],
#)


# syntaxnet ops
tf_custom_op_library(
    name = "parser_ops",
    #srcs = ["//tensorflow/user_ops/syntaxnet/ops/parser_ops.cc"],
    deps = [
        #"//tensorflow/user_ops/syntaxnet:base",
	"//util/utf8:unicodetext",
        "//tensorflow/user_ops/syntaxnet:document_filters",
        "//tensorflow/user_ops/syntaxnet:lexicon_builder",
        "//tensorflow/user_ops/syntaxnet:reader_ops",
        "//tensorflow/user_ops/syntaxnet:unpack_sparse_features",
        #":lib"
    ],
)

# dragnn ops

tf_custom_op_library(
    name = "compute_session_op",
    srcs = ["ops/compute_session_op.h",],
    deps = [
        "//tensorflow/user_ops/dragnn:compute_session",
        "//tensorflow/user_ops/dragnn:resource_container",
        #"//tensorflow/user_ops/syntaxnet:base",
	"//util/utf8:unicodetext",
        "//tensorflow/user_ops/dragnn/protos:data_proto",
        "//tensorflow/user_ops/dragnn/protos:trace_proto",
        "//tensorflow/user_ops/dragnn/protos:spec_proto",
        #"@org_tensorflow//third_party/eigen3",
        #":lib",
        ],
)

tf_custom_op_library(
    name = "dragnn_ops",
    srcs = ["ops/dragnn_op_kernels.cc"],
    deps = [
        "//tensorflow/user_ops/dragnn:compute_session",
        ":compute_session_op",
        "//tensorflow/user_ops/dragnn:compute_session_pool",
        "//tensorflow/user_ops/dragnn:resource_container",
        "//tensorflow/user_ops/dragnn/protos:data_proto",
        "//tensorflow/user_ops/dragnn/protos:spec_proto",
        #"//tensorflow/user_ops/syntaxnet:base",
	"//util/utf8:unicodetext",
        #"//third_party/eigen3",
        #":lib",
        ],
        
)

tf_custom_op_library(
    name = "dragnn_bulk_ops",
    srcs = ["ops/dragnn_bulk_op_kernels.cc", "ops/compute_session_op.h"],
    deps = [
        ":compute_session_op",
        #"//tensorflow/user_ops/syntaxnet:base",
	"//util/utf8:unicodetext",
        "//tensorflow/user_ops/syntaxnet:registry",
        "//tensorflow/user_ops/dragnn:compute_session",
        "//tensorflow/user_ops/dragnn/interfaces:component",
        "//tensorflow/user_ops/dragnn:resource_container",
        "//tensorflow/user_ops/dragnn:index_translator",
        #"@org_tensorflow//third_party/eigen3",
        #":lib",
        ],
)


#tf_custom_op_library(
#    name = "ackermann_op.so",
#    srcs = ["ackermann_op.cc"],
#)

#tf_py_test(
#    name = "ackermann_test",
#    size = "small",
#    srcs = ["ackermann_test.py"],
#    additional_deps = ["//tensorflow:tensorflow_py"],
#    data = [":ackermann_op.so"],
#)

#tf_custom_op_library(
#    name = "duplicate_op.so",
#    srcs = ["duplicate_op.cc"],
#)

#tf_py_test(
#    name = "duplicate_op_test",
#    size = "small",
#    srcs = ["duplicate_op_test.py"],
#    additional_deps = ["//tensorflow:tensorflow_py"],
#    data = [":duplicate_op.so"],
#)

#tf_custom_op_library(
#    name = "invalid_op.so",
#    srcs = ["invalid_op.cc"],
#)

#tf_py_test(
#    name = "invalid_op_test",
#    size = "small",
#    srcs = ["invalid_op_test.py"],
#    additional_deps = ["//tensorflow:tensorflow_py"],
#    data = [":invalid_op.so"],
#)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
