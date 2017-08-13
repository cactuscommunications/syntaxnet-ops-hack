package(default_visibility = ["//visibility:public"])

cc_library(
    name = "sentence_input_batch",
    srcs = ["sentence_input_batch.cc"],
    hdrs = ["sentence_input_batch.h"],
    deps = [
        ":syntaxnet_sentence",
        "@org_tensorflow//tensorflow/core/dragnn/interfaces:input_batch",
        "@org_tensorflow//tensorflow/core/syntaxnet:base",
        "@org_tensorflow//tensorflow/core/syntaxnet:sentence_proto",
    ],
)

cc_library(
    name = "syntaxnet_sentence",
    hdrs = ["syntaxnet_sentence.h"],
    deps = [
        "@org_tensorflow//tensorflow/core/syntaxnet:sentence_proto",
        "@org_tensorflow//tensorflow/core/syntaxnet:workspace",
    ],
)

#cc_test(
#    name = "sentence_input_batch_test",
#    srcs = ["sentence_input_batch_test.cc"],
#    deps = [
#        ":sentence_input_batch",
#        "@org_tensorflow//tensorflow/core/dragnn/test:generic",
#        "@org_tensorflow//tensorflow/core/syntaxnet:sentence_proto",
#        "@org_tensorflow//tensorflow/core/syntaxnet:test_main",
#        "@org_tensorflow//tensorflow/core:test",
#    ],
#)
