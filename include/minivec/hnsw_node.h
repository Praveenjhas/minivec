#pragma once

#include <cstdint>
#include <vector>

namespace minivec {

struct HNSWNode {
    uint64_t id;

    // neighbors[level] = IDs of nodes connected at that level
    std::vector<std::vector<uint64_t>> neighbors;
};

}