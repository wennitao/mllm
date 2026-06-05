// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// LFM2.5-8B-A1B tokenizer. LFM2 uses the same byte-level BPE machinery as Qwen
// (GPT-2 split regex + byte<->unicode map, HF `tokenizer.json` with array-format
// merges, which BPE::initFromSentencePieceJson handles). Only the special tokens
// and the chat template differ, so we reuse qwen3's regex + BPE and swap those.
#pragma once

#include <vector>
#include <unordered_map>

#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"  // reuse qwen3Regex (identical pattern)

namespace mllm::models::lfm2_moe {

struct Lfm2MoeMessage {
  std::string prompt;
  // ChatML with the LFM2 bos token; assistant turn is left open for generation.
  static inline std::string message_template =
      "<|startoftext|><|im_start|>user\n{{{prompt}}}<|im_end|>\n<|im_start|>assistant\n";
};

class Lfm2MoeTokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  explicit Lfm2MoeTokenizer(const std::string& file_path) {
    preprocessor::initLocal();
    preprocessor::makeBytes2UnicodeMap(bytes_2_unicode_dict_);
    for (auto& kv : bytes_2_unicode_dict_) { bytes_2_unicode_dict_inverse_.insert({kv.second, kv.first}); }
    bpe_.initFromSentencePieceJson(file_path);

    // LFM2 special tokens (subset relevant to text generation).
    special_tokens_trie_.add(L"<|pad|>");
    special_tokens_trie_.add(L"<|startoftext|>");
    special_tokens_trie_.add(L"<|endoftext|>");
    special_tokens_trie_.add(L"<|im_start|>");
    special_tokens_trie_.add(L"<|im_end|>");
    special_tokens_trie_.add(L"<|tool_list_start|>");
    special_tokens_trie_.add(L"<|tool_list_end|>");
    special_tokens_trie_.add(L"<|tool_call_start|>");
    special_tokens_trie_.add(L"<|tool_call_end|>");
    special_tokens_trie_.add(L"<think>");
    special_tokens_trie_.add(L"</think>");
  }

  std::vector<std::wstring> _tokenize(const std::string& str) override {
    std::vector<std::wstring> ret;
    std::vector<std::wstring> splitted;
    ::mllm::models::qwen3::qwen3Regex(str, splitted);
    for (const auto& s : splitted) {
      auto utf_8_str = preprocessor::wideString2Utf8String(s);
      std::wstring mapped_str;
      for (unsigned char c : utf_8_str) { mapped_str.push_back(bytes_2_unicode_dict_[c]); }
      auto bpe_ts = bpe_._bpe(mapped_str);
      for (const auto& bpe_t : bpe_ts) { ret.push_back(bpe_t); }
    }
    return ret;
  }

  std::vector<std::wstring> tokenize(const std::string& str) override {
    auto tokens = special_tokens_trie_.split(preprocessor::utf8string2WideString(str));
    std::vector<std::wstring> all_tokens;
    for (const auto& token : tokens) {
      if (special_tokens_trie_.isSpecialToken(token)) {
        all_tokens.emplace_back(token);
        continue;
      }
      auto tmp_tokens = _tokenize(preprocessor::wideString2Utf8String(token));
      all_tokens.insert(all_tokens.end(), tmp_tokens.begin(), tmp_tokens.end());
    }
    return all_tokens;
  }

  std::wstring _detokenize(int64_t pos_idx) override { return bpe_._lookup_inverse_vocab(pos_idx); }

  std::wstring detokenize(int64_t pos_idx) override {
    auto str = _detokenize(pos_idx);
    std::string utf_8_str;
    for (wchar_t c : str) { utf_8_str.push_back((unsigned char)(bytes_2_unicode_dict_inverse_[c])); }
    return {mllm::preprocessor::utf8string2WideString(utf_8_str)};
  }

  Tensor convert2Ids(const std::vector<std::wstring>& strs) override {
    std::vector<int64_t> ids;
    ids.reserve(strs.size());
    for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }
    Tensor ret = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                     .setMemType(kExtraInput)
                     .setName("lfm2-tokenizer-i0")
                     .alloc();
    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }
    return ret;
  }

  ARGenerationOutputPast convertMessage(const Lfm2MoeMessage& message) {
    auto applied_string = Lfm2MoeMessage::message_template;
    size_t pos = applied_string.find("{{{prompt}}}");
    applied_string.replace(pos, 12, message.prompt);

    auto sequence_str = tokenize(applied_string);
    std::vector<int64_t> ids;
    ids.reserve(sequence_str.size());
    for (const auto& str : sequence_str) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    Tensor sequence = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                          .setMemType(kNormal)
                          .setName("lfm2-tokenizer-i0")
                          .alloc();
    auto ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return {{"sequence", sequence}};
  }

  // Tokenize a raw string (no chat template) to ids — useful for validation.
  std::vector<int64_t> encodeRaw(const std::string& str) {
    auto toks = tokenize(str);
    std::vector<int64_t> ids;
    ids.reserve(toks.size());
    for (const auto& t : toks) { ids.emplace_back(bpe_._lookup_vocab(t)); }
    return ids;
  }

 private:
  preprocessor::BPE bpe_;
  std::unordered_map<std::wint_t, wchar_t> bytes_2_unicode_dict_;
  std::unordered_map<wchar_t, std::wint_t> bytes_2_unicode_dict_inverse_;
};

}  // namespace mllm::models::lfm2_moe
