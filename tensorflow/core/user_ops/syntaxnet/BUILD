# Description:
# A syntactic parser and part-of-speech tagger in TensorFlow.

package(
    default_visibility = [
        "//visibility:public",
    ],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_copts"
)

load(
    "//tensorflow/core/user_ops/syntaxnet:syntaxnet.bzl",
    "tf_proto_library",
    "tf_proto_library_py",
    #"tf_gen_op_libs",
    #"tf_gen_op_wrapper_py",
)

#load(
#    "//tensorflow/core:platform/default/build_config.bzl",
#    "tf_proto_library",
#    "tf_proto_library_py",
#    #"tf_gen_op_libs",
#    #"tf_gen_op_wrapper_py",
#)





# proto libraries

tf_proto_library(
    name = "feature_extractor_proto",
    srcs = ["feature_extractor.proto"],
)

tf_proto_library(
    name = "sentence_proto",
    srcs = ["sentence.proto"],
)

tf_proto_library(
    name = "kbest_syntax_proto",
    srcs = ["kbest_syntax.proto"],
    protodeps = [":sentence_proto_cc"],
)

#tf_proto_library_py(
#    name = "sentence_py_pb2",
#    srcs = ["sentence.proto"],
#)

tf_proto_library(
    name = "dictionary_proto",
    srcs = ["dictionary.proto"],
)

#tf_proto_library_py(
#    name = "dictionary_py_pb2",
#    srcs = ["dictionary.proto"],
#)

tf_proto_library(
    name = "task_spec_proto",
    srcs = ["task_spec.proto"],
)

#tf_proto_library_py(
#    name = "task_spec_py_pb2",
#    srcs = ["task_spec.proto"],
#)

tf_proto_library(
    name = "sparse_proto",
    srcs = ["sparse.proto"],
)

#tf_proto_library_py(
#    name = "sparse_py_pb2",
#    srcs = ["sparse.proto"],
#)

# cc libraries for feature extraction and parsing

cc_library(
    name = "base",
    hdrs = ["base.h"],
    visibility = ["//visibility:public"],
    copts= tf_copts(),
    deps = [
        "@com_googlesource_code_re2//:re2",
        "@protobuf_archive//:protobuf",
	#"//tensorflow/core:protos_all",
	#"//tensorflow/core:platform_protobuf",
	#":framework",
	#"tensorflow/core/kernels_reader_base",
        "@org_tensorflow//third_party/eigen3",
	"@org_tensorflow//tensorflow/core:protos_all_cc",
	#"@org_tensorflow//tensorflow/core:core",
	"//tensorflow/core:framework_headers_lib",
	"@org_tensorflow//tensorflow/core:lib",
	#"@org_tensorflow//tensorflow/core:framework"
    ] #+ select({
    #    "//conditions:default": [
    #        "//tensorflow/core:framework",
    #        "//tensorflow/core:lib",
    #    ],
    #    "@org_tensorflow//tensorflow:darwin": [
    #        ,
    #    ],
    #}),
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = [
        "utils.h",
    ],
    deps = [
        ":base",
        "//tensorflow/core/user_ops/util/utf8:unicodetext",
        #"//tensorflow/core:framework",
        #"//tensorflow/core:lib",
    ],
)

cc_library(
    name = "document_format",
    srcs = ["document_format.cc"],
    hdrs = ["document_format.h"],
    deps = [
        ":registry",
        ":sentence_proto_cc",
        ":task_context",
    ],
)

cc_library(
    name = "text_formats",
    srcs = ["text_formats.cc"],
    deps = [
        ":base",
        ":document_format",
        ":segmenter_utils",
        ":sentence_proto_cc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "fml_parser",
    srcs = ["fml_parser.cc"],
    hdrs = ["fml_parser.h"],
    deps = [
	":base",
        ":feature_extractor_proto_cc",
        ":utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "proto_io",
    hdrs = ["proto_io.h"],
    deps = [
        ":feature_extractor_proto_cc",
        ":fml_parser",
        ":sentence_proto_cc",
        ":task_context",
    ],
    alwayslink = 1,
)

cc_library(
    name = "char_properties",
    srcs = ["char_properties.cc"],
    visibility = ["//visibility:public"],
    hdrs = ["char_properties.h"],
    deps = [
	":base",
        ":registry",
        ":utils",
        "//tensorflow/core/user_ops/util/utf8:unicodetext",
	"@com_google_absl//absl/base:core_headers",
	
    ],
    alwayslink = 1,
)

cc_library(
    name = "char_ngram_string_extractor",
    srcs = ["char_ngram_string_extractor.cc"],
    visibility = ["//visibility:public"],
    hdrs = ["char_ngram_string_extractor.h"],
    deps = [
        #":base",
        ":segmenter_utils",
        ":task_context",
    ],
    alwayslink = 1,
)

cc_library(
    name = "segmenter_utils",
    srcs = ["segmenter_utils.cc"],
    visibility = ["//visibility:public"],
    hdrs = ["segmenter_utils.h"],
    copts= tf_copts(),
    deps = [
        ":base",
        ":char_properties",
        ":sentence_proto_cc",
        #":utils",
        "//tensorflow/core/user_ops/third_party/utf:utf",
        "//tensorflow/core/user_ops/util/utf8:unicodetext",
    ],
    alwayslink = 1,
)

cc_library(
    name = "feature_extractor",
    srcs = ["feature_extractor.cc"],
    visibility = ["//visibility:public"],
    hdrs = [
        "feature_extractor.h",
        "feature_types.h",
    ],
    deps = [
        ":document_format",
        ":feature_extractor_proto_cc",
        ":proto_io",
        ":sentence_proto_cc",
        ":task_context",
        ":utils",
        ":workspace",
    ],
    alwayslink = 1,
)

cc_library(
    name = "affix",
    srcs = ["affix.cc"],
    visibility = ["//visibility:public"],
    hdrs = ["affix.h"],
    deps = [
        ":dictionary_proto_cc",
        ":feature_extractor",
        ":sentence_proto_cc",
        ":shared_store",
        ":term_frequency_map",
        ":utils",
        ":workspace",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sentence_features",
    srcs = ["sentence_features.cc"],
    visibility = ["//visibility:public"],
    hdrs = ["sentence_features.h"],
    deps = [
        ":affix",
        ":char_ngram_string_extractor",
        ":char_properties",
        ":feature_extractor",
        ":registry",
        ":segmenter_utils",
        ":shared_store",
        ":task_context",
        ":workspace",
        "//tensorflow/core/user_ops/util/utf8:unicodetext",
    ],
    alwayslink = 1,
)

cc_library(
    name = "whole_sentence_features",
    srcs = ["whole_sentence_features.cc"],
    visibility = ["//visibility:public"],
    hdrs = ["whole_sentence_features.h"],
    deps = [
        ":base",
        ":feature_extractor",
        ":registry",
        ":task_context",
        ":workspace",
    ],
    alwayslink = 1,
)

cc_library(
    name = "shared_store",
    srcs = ["shared_store.cc"],
    visibility = ["//visibility:public"],
    hdrs = ["shared_store.h"],
    deps = [
	":base",
        ":utils",
    ],
)

cc_library(
    name = "registry",
    srcs = ["registry.cc"],
    visibility = ["//visibility:public"],
    hdrs = ["registry.h"],
    deps = [
        ":utils",
	":base",
        #"//tensorflow/core:framework",
    ],
)

cc_library(
    name = "workspace",
    srcs = ["workspace.cc"],
    visibility = ["//visibility:public"],
    hdrs = ["workspace.h"],
    copts= tf_copts(),
    deps = [
        ":utils",
    ],
)

cc_library(
    name = "task_context",
    srcs = ["task_context.cc"],
    visibility = ["//visibility:public"],
    hdrs = ["task_context.h"],
    deps = [
        ":task_spec_proto_cc",
        ":utils",
    ],
)

cc_library(
    name = "term_frequency_map",
    srcs = ["term_frequency_map.cc"],
    hdrs = ["term_frequency_map.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//tensorflow/core:framework_headers_lib",
        #"//tensorflow/core:framework",
        #"//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "morphology_label_set",
    srcs = ["morphology_label_set.cc"],
    visibility = ["//visibility:public"],
    hdrs = ["morphology_label_set.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        #"//tensorflow/core:framework",
        #"//tensorflow/core:lib",
        ":document_format",
        ":feature_extractor",
        ":proto_io",
        ":registry",
        ":sentence_proto_cc",
        ":utils",
    ],
)

cc_library(
    name = "parser_transitions",
    srcs = [
        "arc_standard_transitions.cc",
        "binary_segment_state.cc",
        "binary_segment_transitions.cc",
        "char_shift_transitions.cc",
        "head_transitions.cc",
        "head_transitions.h",
        "label_transitions.cc",
        "label_transitions.h",
        "morpher_transitions.cc",
        "once_transitions.cc",
        "parser_features.cc",
        "parser_state.cc",
        "parser_transitions.cc",
        "shift_transitions.cc",
        "tagger_transitions.cc",
    ],
    visibility = ["//visibility:public"],
    hdrs = [
        "binary_segment_state.h",
        "char_shift_transitions.h",
        "parser_features.h",
        "parser_state.h",
        "parser_transitions.h",
    ],
    deps = [
        ":base",
        ":feature_extractor",
        ":morphology_label_set",
        ":registry",
        ":segmenter_utils",
        ":sentence_features",
        ":sentence_proto_cc",
        ":shared_store",
        ":task_context",
        ":term_frequency_map",
        ":utils",
        ":whole_sentence_features",
        ":workspace",
    ],
    alwayslink = 1,
)

#cc_library(
#    name = "populate_test_inputs",
#    testonly = 1,
#    srcs = ["populate_test_inputs.cc"],
#    hdrs = ["populate_test_inputs.h"],
#    deps = [
#        ":dictionary_proto",
#        ":sentence_proto",
#        ":task_context",
#        ":task_spec_proto",
#        ":term_frequency_map",
#        ":test_main",
#    ],
#)

cc_library(
    name = "embedding_feature_extractor",
    srcs = ["embedding_feature_extractor.cc"],
    visibility = ["//visibility:public"],
    hdrs = ["embedding_feature_extractor.h"],
    deps = [
        ":base",
        ":feature_extractor",
        ":parser_transitions",
        ":sentence_features",
        ":sparse_proto_cc",
        ":task_context",
        ":utils",
        ":workspace",
    ],
)

cc_library(
    name = "sentence_batch",
    srcs = ["sentence_batch.cc"],
    visibility = ["//visibility:public"],
    hdrs = ["sentence_batch.h"],
    deps = [
        ":embedding_feature_extractor",
        ":feature_extractor",
        ":parser_transitions",
        ":sentence_proto_cc",
        ":sparse_proto_cc",
        ":task_context",
        ":task_spec_proto_cc",
        ":term_frequency_map",
        ":workspace",
    ],
)

cc_library(
    name = "reader_ops",
    srcs = [
        "beam_reader_ops.cc",
        "reader_ops.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":parser_transitions",
        ":sentence_batch",
        ":sentence_proto_cc",
        ":sparse_proto_cc",
        ":task_context",
        ":task_spec_proto_cc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "document_filters",
    srcs = ["document_filters.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":document_format",
        ":parser_transitions",
        ":segmenter_utils",
        ":sentence_batch",
        ":sentence_proto_cc",
        ":task_context",
        ":text_formats",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lexicon_builder",
    srcs = ["lexicon_builder.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":affix",
        ":base",
        ":char_ngram_string_extractor",
        ":feature_extractor",
        ":parser_transitions",
        ":segmenter_utils",
        ":sentence_batch",
        ":sentence_proto_cc",
        ":term_frequency_map",
        ":text_formats",
        ":utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "unpack_sparse_features",
    srcs = ["unpack_sparse_features.cc"],
    deps = [
        ":utils",
        #"@org_tensorflow/util/utf8:unicodetext",
        "//tensorflow/core:framework_headers_lib",
        #"//tensorflow/core:framework",
        #"//tensorflow/core:lib",
        ":base",
        ":sparse_proto_cc",
    ],
    alwayslink = 1,
)

#cc_library(
#    name = "parser_ops_cc",
#    srcs = ["ops/parser_ops.cc"],
#    visibility = ["//visibility:public"],
#    deps = [
#        ":base",
#        ":document_filters",
#        ":lexicon_builder",
#        ":reader_ops",
#        ":unpack_sparse_features",
#        "//tensorflow/core:framework",
#        "//tensorflow/core:lib",
#        "//tensorflow/core:framework_headers_lib"
#    ],
#    alwayslink = 1,
#)

#cc_binary(
#    name = "parser_ops.so",
#    linkopts = select({
#        "//conditions:default": ["-lm"],
#        "@org_tensorflow//tensorflow:darwin": [],
#    }),
#    linkshared = 1,
#    linkstatic = 1,
#    deps = [
#        ":parser_ops_cc",
#    ],
#)
#
