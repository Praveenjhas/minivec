#include "minivec/cosine_metric.h"

#include <cmath>
#include <stdexcept>

namespace minivec {

float CosineMetric::calculate(
    const std::vector<float>& a,
    const std::vector<float>& b
) const {

    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vectors must have the same dimension"
        );
    }

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (std::size_t i = 0; i < a.size(); ++i) {

        dot += a[i] * b[i];

        norm_a += a[i] * a[i];

        norm_b += b[i] * b[i];
    }

    if (norm_a == 0.0f || norm_b == 0.0f) {
        throw std::invalid_argument(
            "Cosine distance is undefined for zero vectors"
        );
    }

    float cosine_similarity =
        dot / (std::sqrt(norm_a) * std::sqrt(norm_b));

    return 1.0f - cosine_similarity;
}

}