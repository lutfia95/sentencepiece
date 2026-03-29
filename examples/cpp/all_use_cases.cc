#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"

namespace fs = std::filesystem;

namespace {

using sentencepiece::SentencePieceProcessor;
using sentencepiece::SentencePieceTrainer;
using sentencepiece::util::Status;

constexpr char kRepoRoot[] = SENTENCEPIECE_REPO_ROOT;

void Check(const Status &status, const std::string &context) {
  if (!status.ok()) {
    std::cerr << "[ERROR] " << context << ": " << status.ToString() << '\n';
    std::exit(1);
  }
}

std::string ReadFile(const fs::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    std::cerr << "[ERROR] failed to open " << path << '\n';
    std::exit(1);
  }

  return std::string((std::istreambuf_iterator<char>(stream)),
                     std::istreambuf_iterator<char>());
}

template <typename T>
void PrintVector(const std::vector<T> &values, const std::string &label) {
  std::cout << label << ": [";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) std::cout << ", ";
    std::cout << values[i];
  }
  std::cout << "]\n";
}

void PrintStringVector(const std::vector<std::string> &values,
                       const std::string &label) {
  if (!label.empty()) {
    std::cout << label << ": ";
  }
  std::cout << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) std::cout << ", ";
    std::cout << '"' << values[i] << '"';
  }
  std::cout << "]\n";
}

void PrintHeader(const std::string &title) {
  std::cout << "\n== " << title << " ==\n";
}

fs::path PrepareOutputDir() {
  const fs::path output_dir = fs::current_path() / "sentencepiece_cpp_example_output";
  fs::create_directories(output_dir);
  return output_dir;
}

fs::path TrainModelFromCorpusFile(const fs::path &output_dir) {
  PrintHeader("Train From Corpus File");

  const fs::path corpus = fs::path(kRepoRoot) / "data" / "botchan.txt";
  const fs::path model_prefix = output_dir / "botchan_unigram";

  sentencepiece::SetDataDir((fs::path(kRepoRoot) / "data").string());

  const std::string args =
      "--input=" + corpus.string() +
      " --model_prefix=" + model_prefix.string() +
      " --model_type=unigram"
      " --vocab_size=1000"
      " --character_coverage=1.0"
      " --user_defined_symbols=<cls>,<sep>";

  Check(SentencePieceTrainer::Train(args),
        "training unigram model from corpus file");

  std::cout << "model: " << model_prefix.string() << ".model\n";
  std::cout << "vocab: " << model_prefix.string() << ".vocab\n";
  return model_prefix.string() + ".model";
}

void DemoLoadEncodeDecode(const fs::path &model_file) {
  PrintHeader("Load, Encode, Decode");

  SentencePieceProcessor processor;
  Check(processor.Load(model_file.string()), "loading model from file");

  const std::string input = "SentencePiece can encode raw text.";
  std::vector<std::string> pieces;
  std::vector<int> ids;

  Check(processor.Encode(input, &pieces), "encoding input to pieces");
  Check(processor.Encode(input, &ids), "encoding input to ids");

  PrintStringVector(pieces, "pieces");
  PrintVector(ids, "ids");

  std::string decoded_from_pieces;
  std::string decoded_from_ids;
  Check(processor.Decode(pieces, &decoded_from_pieces),
        "decoding from pieces");
  Check(processor.Decode(ids, &decoded_from_ids), "decoding from ids");

  std::cout << "decoded_from_pieces: " << decoded_from_pieces << '\n';
  std::cout << "decoded_from_ids: " << decoded_from_ids << '\n';
}

void DemoNBestAndSampling(const fs::path &model_file) {
  PrintHeader("NBest And Sampling");

  SentencePieceProcessor processor;
  Check(processor.Load(model_file.string()), "loading model for sampling");

  const std::string input = "New York";

  std::vector<std::vector<std::string>> nbest_pieces;
  Check(processor.NBestEncode(input, 3, &nbest_pieces), "n-best encode");
  for (size_t i = 0; i < nbest_pieces.size(); ++i) {
    PrintStringVector(nbest_pieces[i], "nbest[" + std::to_string(i) + "]");
  }

  std::vector<std::string> sampled_pieces;
  Check(processor.SampleEncode(input, -1, 0.1f, &sampled_pieces),
        "sample encode");
  PrintStringVector(sampled_pieces, "sampled_pieces");

  auto samples =
      processor.SampleEncodeAndScoreAsPieces(input, 3, 0.1f, true, true);
  for (size_t i = 0; i < samples.size(); ++i) {
    std::cout << "sample_with_score[" << i << "]: score=" << std::fixed
              << std::setprecision(6) << samples[i].second << " pieces=";
    PrintStringVector(samples[i].first, "");
  }

  std::cout << "entropy: " << processor.CalculateEntropy(input, 0.1f) << '\n';
}

void DemoImmutableProto(const fs::path &model_file) {
  PrintHeader("Immutable Proto");

  SentencePieceProcessor processor;
  Check(processor.Load(model_file.string()), "loading model for proto demo");

  const auto proto =
      processor.EncodeAsImmutableProto("This example shows byte offsets.");

  std::cout << "text: " << proto.text() << '\n';
  for (const auto &piece : proto.pieces()) {
    std::cout << "piece=" << piece.piece()
              << " id=" << piece.id()
              << " begin=" << piece.begin()
              << " end=" << piece.end()
              << " surface=\"" << piece.surface() << "\"\n";
  }
}

void DemoVocabularyHelpers(const fs::path &model_file) {
  PrintHeader("Vocabulary Helpers");

  SentencePieceProcessor processor;
  Check(processor.Load(model_file.string()), "loading model for vocab demo");

  std::cout << "piece_size: " << processor.GetPieceSize() << '\n';
  std::cout << "bos id: " << processor.bos_id() << '\n';
  std::cout << "eos id: " << processor.eos_id() << '\n';
  std::cout << "unk id: " << processor.unk_id() << '\n';

  const std::string piece = "<sep>";
  const int piece_id = processor.PieceToId(piece);
  std::cout << "PieceToId(\"" << piece << "\"): " << piece_id << '\n';
  std::cout << "IdToPiece(" << piece_id << "): "
            << processor.IdToPiece(piece_id) << '\n';
  std::cout << "IsUnknown(" << processor.unk_id() << "): "
            << processor.IsUnknown(processor.unk_id()) << '\n';
  std::cout << "IsControl(" << processor.bos_id() << "): "
            << processor.IsControl(processor.bos_id()) << '\n';
}

void DemoExtraOptions(const fs::path &model_file) {
  PrintHeader("Extra Encode And Decode Options");

  SentencePieceProcessor processor;
  Check(processor.Load(model_file.string()), "loading model for option demo");
  Check(processor.SetEncodeExtraOptions("reverse:bos:eos"),
        "setting encode extra options");

  std::vector<std::string> pieces;
  Check(processor.Encode("extra options demo", &pieces),
        "encoding with reverse/bos/eos");
  PrintStringVector(pieces, "pieces_with_reverse_bos_eos");

  SentencePieceProcessor reverse_processor;
  Check(reverse_processor.Load(model_file.string()),
        "loading model for reverse decode");
  Check(reverse_processor.SetDecodeExtraOptions("reverse"),
        "setting reverse decode option");

  std::string reversed;
  Check(reverse_processor.Decode(pieces, &reversed),
        "decoding with reverse option");
  std::cout << "reversed_decoded_text: " << reversed << '\n';
}

void DemoLoadSerializedModel(const fs::path &model_file) {
  PrintHeader("Load Serialized Model");

  const std::string model_blob = ReadFile(model_file);
  SentencePieceProcessor processor;
  Check(processor.LoadFromSerializedProto(model_blob),
        "loading serialized model proto");

  PrintStringVector(processor.EncodeAsPieces("serialized model load works"),
                    "pieces");
}

void DemoTrainInMemory() {
  PrintHeader("Train From In-Memory Sentences");

  std::vector<std::string> sentences = {
      "SentencePiece can train from strings.",
      "This avoids depending on local corpus files.",
      "Small demos are useful for tests and examples.",
      "Subword tokenization works on raw text."};

  std::string model_proto;
  sentencepiece::SetDataDir((fs::path(kRepoRoot) / "data").string());
  Check(SentencePieceTrainer::Train(
            "--model_type=unigram --vocab_size=48 "
            "--character_coverage=1.0 --hard_vocab_limit=false",
            sentences, &model_proto),
        "training model from in-memory sentence list");

  SentencePieceProcessor processor;
  Check(processor.LoadFromSerializedProto(model_proto),
        "loading in-memory trained model");

  PrintStringVector(processor.EncodeAsPieces("raw text example"),
                    "in_memory_pieces");
}

}  // namespace

int main() {
  const fs::path output_dir = PrepareOutputDir();
  const fs::path model_file = TrainModelFromCorpusFile(output_dir);

  DemoLoadEncodeDecode(model_file);
  DemoNBestAndSampling(model_file);
  DemoImmutableProto(model_file);
  DemoVocabularyHelpers(model_file);
  DemoExtraOptions(model_file);
  DemoLoadSerializedModel(model_file);
  DemoTrainInMemory();

  std::cout << "\nArtifacts written to: " << output_dir << '\n';
  return 0;
}
