#pragma once

#include "minivec/vector_record.h"
#include <cstdint>

namespace minivec {

class VectorProvider {
public:
    virtual ~VectorProvider() = default;

    virtual const VectorRecord* get(uint64_t id) const = 0;
};

}