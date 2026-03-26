// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cwchar>
#include <unordered_map>
#include <string>
#include <locale>
#include <cwctype>
#include <iostream>

#include "mllm/utils/Common.hpp"

namespace mllm::preprocessor {

inline std::wint_t ord(const wchar_t* str) { return static_cast<std::wint_t>(*str); }

inline wchar_t chr(std::wint_t value) { return static_cast<wchar_t>(value); }

// some OS has no en_US.UTF-8 but has C.UTF-8.
inline void initLocal(const std::string& local_name = "en_US.UTF-8") {
  const std::string candidates[] = {local_name, "C.UTF-8", "C"};
  for (const auto& candidate : candidates) {
    try {
      auto locale = std::locale(candidate.c_str());
      std::locale::global(locale);
      std::wcout.imbue(locale);
      return;
    } catch (const std::exception&) {
      MLLM_WARN("Failed to set locale to {}", candidate);
    }
  }

  auto classic = std::locale::classic();
  std::locale::global(classic);
  std::wcout.imbue(classic);
}

inline bool isLetter(wchar_t c) { return std::iswalpha(c); }

inline bool isDigit(wchar_t c) { return std::iswdigit(c); }

std::string wideString2Utf8String(const std::wstring& wstr);

std::wstring utf8string2WideString(const std::string& str);

// This function is used for GPT2 Like Tokenizers
//
// same with gpt2.bytes_to_unicode
//
// same with qwen2.bytes_to_unicode
//
/*
Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to
whitespace/control characters the bpe code barfs on.

The reversible bpe codes work on unicode strings. This means you need a large # of unicode
characters in your vocab if you want to avoid UNKs. When you're at something like a 10B token
dataset you end up needing around 5K for decent coverage. This is a significant percentage of
your normal, say, 32K bpe vocab. To avoid that, we want lookup tables between utf-8 bytes and
unicode strings.
*/
void makeBytes2UnicodeMap(std::unordered_map<std::wint_t, wchar_t>& dict);

}  // namespace mllm::preprocessor
