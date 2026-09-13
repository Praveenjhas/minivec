#include "minivec/vector_store.h"

namespace minivec {

void VectorStore::insert(const VectorRecord& record) {

    records_[record.id] = record;
}


void VectorStore::remove(uint64_t id) {

    records_.erase(id);
}


const VectorRecord* VectorStore::get(uint64_t id) const {

    auto it = records_.find(id);

    if (it == records_.end()) {
        return nullptr;
    }

    return &it->second;
}
const std::unordered_map<uint64_t, VectorRecord>&
VectorStore::records() const {
    return records_;
}


std::size_t VectorStore::size() const {

    return records_.size();
}

} // namespace minivec