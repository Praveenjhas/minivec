#include "minivec/collection.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "minivec/filter.h"
#include "minivec/hnsw_index.h"
#include "minivec/rrf.h"

namespace minivec {

// ============================================================
// Normal constructor
// ============================================================

Collection::Collection(
    std::size_t dimension,
    std::unique_ptr<DistanceMetric> metric,
    std::unique_ptr<IndexFactory> index_factory
)
    : dimension_(dimension),
      store_(std::make_unique<VectorStore>()),
      metric_(std::move(metric)),
      index_(nullptr),
      keyword_index_() {

    if (dimension_ == 0) {
        throw std::invalid_argument(
            "Collection dimension must be greater than zero"
        );
    }

    if (!metric_) {
        throw std::invalid_argument(
            "Distance metric cannot be null"
        );
    }

    if (!index_factory) {
        throw std::invalid_argument(
            "Index factory cannot be null"
        );
    }

    index_ =
        index_factory->create(
            *store_,
            *metric_
        );

    if (!index_) {
        throw std::runtime_error(
            "Index factory returned null index"
        );
    }
}


// ============================================================
// Persistence constructor
// ============================================================

Collection::Collection(
    PersistenceTag,
    std::size_t dimension,
    std::unique_ptr<VectorStore> store,
    std::unique_ptr<DistanceMetric> metric,
    std::unique_ptr<VectorIndex> index
)
    : dimension_(dimension),
      store_(std::move(store)),
      metric_(std::move(metric)),
      index_(std::move(index)),
      keyword_index_() {

    if (dimension_ == 0) {
        throw std::invalid_argument(
            "Collection dimension must be greater than zero"
        );
    }

    if (!store_) {
        throw std::invalid_argument(
            "Vector store cannot be null"
        );
    }

    if (!metric_) {
        throw std::invalid_argument(
            "Distance metric cannot be null"
        );
    }

    if (!index_) {
        throw std::invalid_argument(
            "Vector index cannot be null"
        );
    }

    // --------------------------------------------------------
    // Rebuild keyword index from persisted records.
    //
    // HNSW is restored directly from persistence.
    // BM25 is derived from persisted text.
    // --------------------------------------------------------

    for (const auto& [id, record] :
         store_->records()) {

        keyword_index_.insert(
            id,
            record.text
        );
    }
}


// ============================================================
// INSERT / UPSERT
// ============================================================

void Collection::insert(
    const VectorRecord& record
) {
    // Writer lock.
    std::unique_lock<std::shared_mutex> lock(
        mutex_
    );

    if (record.values.size() != dimension_) {
        throw std::invalid_argument(
            "Vector dimension does not match collection dimension"
        );
    }

    const VectorRecord* existing =
        store_->get(record.id);

    // --------------------------------------------------------
    // New vector
    // --------------------------------------------------------

    if (existing == nullptr) {

        store_->insert(record);

        index_->insert(record.id);

        keyword_index_.insert(
            record.id,
            record.text
        );

        return;
    }

    // --------------------------------------------------------
    // Existing ID -> upsert
    // --------------------------------------------------------

    index_->remove(record.id);

    keyword_index_.remove(record.id);

    store_->insert(record);

    index_->insert(record.id);

    keyword_index_.insert(
        record.id,
        record.text
    );
}


// ============================================================
// REMOVE
// ============================================================

void Collection::remove(
    uint64_t id
) {
    std::unique_lock<std::shared_mutex> lock(
        mutex_
    );

    index_->remove(id);

    keyword_index_.remove(id);

    store_->remove(id);
}


// ============================================================
// GET
// ============================================================

std::optional<VectorRecord> Collection::get(
    uint64_t id
) const {

    std::shared_lock<std::shared_mutex> lock(
        mutex_
    );

    const VectorRecord* record =
        store_->get(id);

    if (record == nullptr) {
        return std::nullopt;
    }

    // Return a copy.
    //
    // This is safer than returning the internal pointer because
    // the shared lock is released when this function returns.

    return *record;
}


// ============================================================
// VECTOR SEARCH
// ============================================================

std::vector<SearchResult> Collection::search(
    const std::vector<float>& query,
    std::size_t k,
    const std::optional<FilterCondition>& filter
) const {

    std::shared_lock<std::shared_mutex> lock(
        mutex_
    );

    if (query.size() != dimension_) {
        throw std::invalid_argument(
            "Query dimension does not match collection dimension"
        );
    }

    if (k == 0) {
        return {};
    }

    // --------------------------------------------------------
    // No filter
    // --------------------------------------------------------

    if (!filter.has_value()) {

        return index_->search(
            query,
            k
        );
    }

    // --------------------------------------------------------
    // Filtered vector search
    //
    // Search more candidates first because some of them may be
    // removed by the metadata filter.
    // --------------------------------------------------------

    std::size_t candidate_k =
        std::max(
            k * 10,
            k
        );

    auto candidates =
        index_->search(
            query,
            candidate_k
        );

    std::vector<SearchResult> filtered;

    filtered.reserve(
        candidates.size()
    );

    for (const auto& result :
         candidates) {

        const VectorRecord* record =
            store_->get(result.id);

        if (record == nullptr) {
            continue;
        }

        if (matches_filter(
                *record,
                *filter
            )) {

            filtered.push_back(result);
        }
    }

    if (filtered.size() > k) {
        filtered.resize(k);
    }

    return filtered;
}


// ============================================================
// KEYWORD / BM25 SEARCH
// ============================================================

std::vector<SearchResult>
Collection::keyword_search(
    const std::string& query,
    std::size_t k
) const {

    std::shared_lock<std::shared_mutex> lock(
        mutex_
    );

    if (k == 0) {
        return {};
    }

    return keyword_index_.search(
        query,
        k
    );
}


// ============================================================
// HYBRID SEARCH
//
// Vector search
//      +
// BM25 keyword search
//      +
// Metadata filtering
//      +
// Reciprocal Rank Fusion
//
// filter == nullptr -> no metadata filter
// ============================================================

std::vector<SearchResult>
Collection::hybrid_search(
    const std::vector<float>& query_vector,
    const std::string& query_text,
    std::size_t k,
    const FilterCondition* filter
) const {

    if (k == 0) {
        return {};
    }

    // --------------------------------------------------------
    // Validate query dimension
    // --------------------------------------------------------

    if (query_vector.size() != dimension_) {
        throw std::invalid_argument(
            "Query dimension does not match collection dimension"
        );
    }

    std::shared_lock<std::shared_mutex> lock(
        mutex_
    );

    // --------------------------------------------------------
    // Candidate count
    //
    // If filtering is used, retrieving only k candidates can
    // result in fewer than k valid results after filtering.
    //
    // We therefore over-fetch.
    //
    // Example:
    //
    // k = 5
    // candidate_k = 100
    //
    // Search 100 -> filter -> RRF -> return top 5.
    // --------------------------------------------------------

    std::size_t candidate_k =
        std::max(
            k * 10,
            static_cast<std::size_t>(100)
        );

    // --------------------------------------------------------
    // Vector retrieval
    // --------------------------------------------------------

    auto vector_results =
        index_->search(
            query_vector,
            candidate_k
        );

    // --------------------------------------------------------
    // BM25 retrieval
    // --------------------------------------------------------

    auto keyword_results =
        keyword_index_.search(
            query_text,
            candidate_k
        );

    // --------------------------------------------------------
    // Apply metadata filter
    // --------------------------------------------------------

    if (filter != nullptr) {

        auto apply_filter =
            [&](std::vector<SearchResult>& results) {

                results.erase(
                    std::remove_if(
                        results.begin(),
                        results.end(),
                        [&](const SearchResult& result) {

                            const VectorRecord* record =
                                store_->get(result.id);

                            // If the record no longer exists,
                            // remove it from the result list.

                            if (record == nullptr) {
                                return true;
                            }

                            return !matches_filter(
                                *record,
                                *filter
                            );
                        }
                    ),
                    results.end()
                );
            };

        apply_filter(
            vector_results
        );

        apply_filter(
            keyword_results
        );
    }

    // --------------------------------------------------------
    // Reciprocal Rank Fusion
    //
    // Important:
    //
    // RRF does NOT care about the actual vector distance or
    // BM25 score. It uses the rank in each result list.
    // --------------------------------------------------------

    RRF rrf(60);

    return rrf.fuse(
        vector_results,
        keyword_results,
        k
    );
}


// ============================================================
// SIZE
// ============================================================

std::size_t Collection::size() const {

    std::shared_lock<std::shared_mutex> lock(
        mutex_
    );

    return store_->size();
}


// ============================================================
// DIMENSION
// ============================================================

std::size_t Collection::dimension() const {

    // dimension_ never changes after construction.

    return dimension_;
}


// ============================================================
// SET EF SEARCH
// ============================================================

void Collection::set_ef_search(
    std::size_t ef_search
) {

    std::unique_lock<std::shared_mutex> lock(
        mutex_
    );

    HNSWIndex* hnsw =
        dynamic_cast<HNSWIndex*>(
            index_.get()
        );

    if (hnsw == nullptr) {

        throw std::runtime_error(
            "set_ef_search is only supported for HNSW index"
        );
    }

    hnsw->set_ef_search(
        ef_search
    );
}

} // namespace minivec