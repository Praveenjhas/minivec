#pragma once

#include "minivec/index_factory.h"

#include <cstddef>

namespace minivec {

class HNSWIndexFactory : public IndexFactory {
public:

    HNSWIndexFactory(
        std::size_t M = 16,
        std::size_t ef_construction = 200,
        std::size_t ef_search = 50
    );

    std::unique_ptr<VectorIndex> create(
        const VectorProvider& provider,
        const DistanceMetric& metric
    ) const override;

private:

    std::size_t M_;
    std::size_t ef_construction_;
    std::size_t ef_search_;
};

}