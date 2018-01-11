licenses(["notice"])

# Requires --copt -funsigned-char when compiling (unsigned chars).

cc_library(
    name = "unicodetext",
    srcs = [
        "unicodetext.cc",
        "unilib.cc",
    ],
    hdrs = [
        "unicodetext.h",
        "unilib.h",
        "unilib_utf8_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/utf",
	"//tensorflow/user_ops/syntaxnet:base",
        #"//tensorflow/core:framework_headers_lib",
        #"//tensorflow/core:framework",
        #"//tensorflow/core:lib",
    ],
)

cc_test(
    name = "unicodetext_unittest",
    srcs = [
        "gtest_main.cc",
        "unicodetext_unittest.cc",
    ],
    deps = [
        "//tensorflow/core:testlib",
        ":unicodetext",
    ],
)

cc_binary(
    name = "unicodetext_main",
    srcs = ["unicodetext_main.cc"],
    deps = [":unicodetext"],
)
