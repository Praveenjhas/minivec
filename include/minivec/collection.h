#pragma once

#include <optional>

#include "minivec/rrf.h"
#include "minivec/vector_store.h"
#include "minivec/vector_index.h"
#include "minivec/distance_metric.h"
#include "minivec/index_factory.h"
#include "minivec/filter.h"
#include "minivec/keyword_index.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

namespace minivec {

class Persistence;

class Collection {

public:

    Collection(
        std::size_t dimension,
        std::unique_ptr<DistanceMetric> metric,
        std::unique_ptr<IndexFactory> index_factory
    );

    void insert(
        const VectorRecord& record
    );

    std::optional<VectorRecord> get(
        uint64_t id
    ) const;

    void remove(
        uint64_t id
    );

    // --------------------------------------------------------
    // Hybrid search
    //
    // Vector search + BM25 + RRF
    //
    // filter == nullptr means no metadata filtering.
    // --------------------------------------------------------

    std::vector<SearchResult> hybrid_search(
        const std::vector<float>& query_vector,
        const std::string& query_text,
        std::size_t k,
        const FilterCondition* filter = nullptr
    ) const;

    // --------------------------------------------------------
    // Vector search
    // --------------------------------------------------------

    std::vector<SearchResult> search(
        const std::vector<float>& query,
        std::size_t k,
        const std::optional<FilterCondition>& filter = std::nullopt
    ) const;

    // --------------------------------------------------------
    // Keyword / BM25 search
    // --------------------------------------------------------

    std::vector<SearchResult> keyword_search(
        const std::string& query,
        std::size_t k
    ) const;

    std::size_t size() const;

    std::size_t dimension() const;

    void set_ef_search(
        std::size_t ef_search
    );

private:

    friend class Persistence;

    // --------------------------------------------------------
    // Used only by Persistence when loading from disk.
    // --------------------------------------------------------

    struct PersistenceTag {};

    Collection(
        PersistenceTag,
        std::size_t dimension,
        std::unique_ptr<VectorStore> store,
        std::unique_ptr<DistanceMetric> metric,
        std::unique_ptr<VectorIndex> index
    );

    // --------------------------------------------------------
    // Collection state
    // --------------------------------------------------------

    std::size_t dimension_;

    std::unique_ptr<VectorStore> store_;

    std::unique_ptr<DistanceMetric> metric_;

    std::unique_ptr<VectorIndex> index_;

    KeywordIndex keyword_index_;

    // --------------------------------------------------------
    // Concurrency
    // --------------------------------------------------------

    mutable std::shared_mutex mutex_;
};

}