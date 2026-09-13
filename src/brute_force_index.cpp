#include "minivec/brute_force_index.h"

#include <algorithm>

namespace minivec {

BruteForceIndex::BruteForceIndex(
    const VectorProvider& provider,
    const DistanceMetric& metric
)
    : provider_(provider),
      metric_(metric) {
}

void BruteForceIndex::insert(uint64_t id) {
    ids_.insert(id);
}

void BruteForceIndex::remove(uint64_t id) {
    ids_.erase(id);
}

std::vector<SearchResult> BruteForceIndex::search(
    const std::vector<float>& query,
    std::size_t k
) const {

    if (k == 0) {
        return {};
    }

    std::vector<SearchResult> results;

    results.reserve(ids_.size());

    for (uint64_t id : ids_) {

        const VectorRecord* record =
            provider_.get(id);

        if (record == nullptr) {
            continue;
        }

        float distance =
            metric_.calculate(query, record->values);

        results.push_back({id, distance});
    }

    if (results.size() > k) {

        std::partial_sort(
            results.begin(),
            results.begin() + k,
            results.end(),
            [](const SearchResult& a, const SearchResult& b) {
                return a.score < b.score;
            }
        );

        results.resize(k);

    } else {

        std::sort(
            results.begin(),
            results.end(),
            [](const SearchResult& a, const SearchResult& b) {
                return a.score < b.score;
            }
        );
    }

    return results;
}

}