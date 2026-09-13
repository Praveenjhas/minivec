#pragma once

#include <vector>

namespace minivec {

class DistanceMetric {
public:
    virtual ~DistanceMetric() = default;

    virtual float calculate(
        const std::vector<float>& a,
        const std::vector<float>& b
    ) const = 0;
};

}