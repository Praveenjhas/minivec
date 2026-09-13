#pragma once

#include "minivec/metadata.h"

#include <cstdint>
#include <string>
#include <vector>

namespace minivec {

struct VectorRecord {

    uint64_t id;

    std::vector<float> values;

    std::string text;

    Metadata metadata;
};

}