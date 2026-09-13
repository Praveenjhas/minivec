#pragma once

#include "minivec/vector_index.h"
#include "minivec/vector_provider.h"
#include "minivec/distance_metric.h"

#include <cstddef>
#include <memory>

namespace minivec {

class IndexFactory {
public:

    virtual ~IndexFactory() = default;

    virtual std::unique_ptr<VectorIndex> create(
        const VectorProvider& provider,
        const DistanceMetric& metric
    ) const = 0;
};

}