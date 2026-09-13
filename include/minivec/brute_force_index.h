#pragma once

#include "minivec/vector_index.h"
#include "minivec/vector_provider.h"
#include "minivec/distance_metric.h"

#include <cstdint>
#include <unordered_set>

namespace minivec {

class BruteForceIndex : public VectorIndex {
public:
    BruteForceIndex(
        const VectorProvider& provider,
        const DistanceMetric& metric
    );

    void insert(uint64_t id) override;

    void remove(uint64_t id) override;

    std::vector<SearchResult> search(
        const std::vector<float>& query,
        std::size_t k
    ) const override;

private:
    const VectorProvider& provider_;
    const DistanceMetric& metric_;

    std::unordered_set<uint64_t> ids_;
};

}