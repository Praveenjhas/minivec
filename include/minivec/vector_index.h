#pragma once

#include "minivec/search_result.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace minivec {

class VectorIndex {
public:
    virtual ~VectorIndex() = default;

    virtual void insert(uint64_t id) = 0;

    virtual void remove(uint64_t id) = 0;

    virtual std::vector<SearchResult> search(
        const std::vector<float>& query,
        std::size_t k
    ) const = 0;
};

}