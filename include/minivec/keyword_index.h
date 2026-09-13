#pragma once

#include "minivec/search_result.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace minivec {

class KeywordIndex {
public:
    KeywordIndex(
        double k1 = 1.2,
        double b = 0.75
    );

    void insert(
        uint64_t id,
        const std::string& text
    );

    void remove(uint64_t id);

    std::vector<SearchResult> search(
        const std::string& query,
        std::size_t k
    ) const;

    std::size_t document_count() const;

    std::size_t total_terms() const;

    double average_document_length() const;

private:
    using PostingList =
        std::unordered_map<uint64_t, std::size_t>;

    static std::vector<std::string> tokenize(
        const std::string& text
    );

    // term -> (document ID -> term frequency)
    std::unordered_map<
        std::string,
        PostingList
    > inverted_index_;

    // document ID -> number of tokens
    std::unordered_map<
        uint64_t,
        std::size_t
    > document_lengths_;

    std::size_t total_documents_ = 0;

    std::size_t total_terms_ = 0;

    // BM25 parameters
    double k1_;

    double b_;
};

}