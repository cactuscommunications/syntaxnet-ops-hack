package(default_visibility = ["//visibility:public"])

cc_library(
    name = "sentence_input_batch",
    srcs = ["sentence_input_batch.cc"],
    hdrs = ["sentence_input_batch.h"],
    deps = [
        ":syntaxnet_sentence",
        "//tensorflow/core/dragnn/interfaces:input_batch",
        "//tensorflow/core/syntaxnet:base",
        "//tensorflow/core/syntaxnet:sentence_proto",
    ],
)

cc_library(
    name = "syntaxnet_sentence",
    hdrs = ["syntaxnet_sentence.h"],
    deps = [
        "//tensorflow/core/syntaxnet:sentence_proto",
        "//tensorflow/core/syntaxnet:workspace",
    ],
)

#cc_test(
#    name = "sentence_input_batch_test",
#    srcs = ["sentence_input_batch_test.cc"],
#    deps = [
#        ":sentence_input_batch",
#        "//tensorflow/core/dragnn/test:generic",
#        "//tensorflow/core/syntaxnet:sentence_proto",
#        "//tensorflow/core/syntaxnet:test_main",
#        "//tensorflow/core:test",
#    ],
#)
