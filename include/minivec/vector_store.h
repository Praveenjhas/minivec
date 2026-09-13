#pragma once

#include "minivec/vector_record.h"
#include "minivec/vector_provider.h"

#include <cstdint>
#include <cstddef>
#include <unordered_map>

namespace minivec {

class VectorStore : public VectorProvider{
public:

    void insert(const VectorRecord& record);

    void remove(uint64_t id);

    const VectorRecord* get(uint64_t id) const override;;

    std::size_t size() const;
   const std::unordered_map<uint64_t, VectorRecord>&
    records() const;

private:

    std::unordered_map<uint64_t, VectorRecord> records_;
};

} // namespace minivec