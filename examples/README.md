# Examples

This directory contains runnable examples that demonstrate common SentencePiece
workflows end to end.

## C++

`cpp/all_use_cases.cc` demonstrates:

- training from a corpus file
- training directly from an in-memory sentence list
- loading a model from disk
- loading a model from a serialized model proto
- encoding to pieces and ids
- decoding from pieces and ids
- n-best encoding
- sampling and sampling with scores
- immutable proto output with byte offsets
- vocabulary inspection helpers
- encode/decode extra options

### Build

```sh
cmake -S . -B build
cmake --build build --target sentencepiece_cpp_all_use_cases
```

### Run

```sh
./build/examples/sentencepiece_cpp_all_use_cases
```

The example writes demo model files under the current working directory in
`sentencepiece_cpp_example_output/`.
