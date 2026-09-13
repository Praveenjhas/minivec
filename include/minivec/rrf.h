#pragma once

#include "minivec/search_result.h"

#include <cstddef>
#include <vector>

namespace minivec {

class RRF {
public:
    explicit RRF(std::size_t k = 60);

    std::vector<SearchResult> fuse(
        const std::vector<SearchResult>& vector_results,
        const std::vector<SearchResult>& keyword_results,
        std::size_t top_k
    ) const;

private:
    std::size_t k_;
};

}