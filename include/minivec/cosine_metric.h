#pragma once

#include "minivec/distance_metric.h"

namespace minivec {

class CosineMetric : public DistanceMetric {
public:
    float calculate(
        const std::vector<float>& a,
        const std::vector<float>& b
    ) const override;
};

}