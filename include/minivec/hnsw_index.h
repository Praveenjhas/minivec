#pragma once

#include "minivec/vector_index.h"
#include "minivec/vector_provider.h"
#include "minivec/distance_metric.h"
#include "minivec/hnsw_node.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>
#include <queue>
#include <unordered_set>
namespace minivec {

class HNSWIndex : public VectorIndex {
public:
    HNSWIndex(
        const VectorProvider& provider,
        const DistanceMetric& metric,
        std::size_t M = 16,
        std::size_t ef_construction = 200,
        std::size_t ef_search = 50
    );

    void insert(uint64_t id) override;
    std::vector<uint64_t> select_neighbors(
    uint64_t node_id,
    const std::vector<SearchResult>& candidates,
    std::size_t max_neighbors
) const;
    
    void connect_nodes(
    uint64_t a,
    uint64_t b,
    int layer
);
     void disconnect_nodes(
    uint64_t a,
    uint64_t b,
    int layer
);
    void remove(uint64_t id) override;
    void print_stats() const;
   uint64_t entry_point() const;
   
   std::size_t max_neighbors_for_layer(
    int layer
) const;

    std::vector<SearchResult> search(
        const std::vector<float>& query,
        std::size_t k
    ) const override;
   int max_level() const;
   void set_ef_search(std::size_t ef_search);
   bool validate_graph() const;
   const std::unordered_map<uint64_t, HNSWNode>&
    nodes() const;

    std::size_t M() const;
    std::size_t ef_construction() const;

    std::size_t ef_search() const;
    void load_state(
    std::unordered_map<uint64_t, HNSWNode> nodes,
    uint64_t entry_point,
    int max_level
);
private:
    uint64_t greedy_search_layer(
    const std::vector<float>& query,
    uint64_t entry_point,
    int layer
) const; 
 std::vector<SearchResult> search_layer(
    const std::vector<float>& query,
    uint64_t entry_point,
    std::size_t ef,
    int layer
) const;

    const VectorProvider& provider_;
    const DistanceMetric& metric_;
    void prune_connections(
    uint64_t node_id,
    int layer
);

    std::size_t M_;
    std::size_t ef_construction_;
    std::size_t ef_search_;

    std::unordered_map<uint64_t, HNSWNode> nodes_;

    uint64_t entry_point_;
    int max_level_;

    std::mt19937 rng_;

    int random_level();

    float distance(
        const std::vector<float>& a,
        const std::vector<float>& b
    ) const;
};

}