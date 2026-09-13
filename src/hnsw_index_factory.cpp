#include "minivec/hnsw_index_factory.h"
#include "minivec/hnsw_index.h"

#include <stdexcept>
#include <utility>

namespace minivec {

HNSWIndexFactory::HNSWIndexFactory(
    std::size_t M,
    std::size_t ef_construction,
    std::size_t ef_search
)
    : M_(M),
      ef_construction_(ef_construction),
      ef_search_(ef_search) {

    if (M_ < 2) {
        throw std::invalid_argument(
            "HNSW M must be at least 2"
        );
    }

    if (ef_construction_ == 0) {
        throw std::invalid_argument(
            "efConstruction must be greater than zero"
        );
    }

    if (ef_search_ == 0) {
        throw std::invalid_argument(
            "efSearch must be greater than zero"
        );
    }
}


std::unique_ptr<VectorIndex>
HNSWIndexFactory::create(
    const VectorProvider& provider,
    const DistanceMetric& metric
) const {

    return std::make_unique<HNSWIndex>(
        provider,
        metric,
        M_,
        ef_construction_,
        ef_search_
    );
}

}