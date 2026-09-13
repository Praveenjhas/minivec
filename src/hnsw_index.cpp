#include "minivec/hnsw_index.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace minivec {

HNSWIndex::HNSWIndex(
    const VectorProvider& provider,
    const DistanceMetric& metric,
    std::size_t M,
    std::size_t ef_construction,
    std::size_t ef_search
)
    : provider_(provider),
      metric_(metric),
      M_(M),
      ef_construction_(ef_construction),
      ef_search_(ef_search),
      entry_point_(0),
      max_level_(-1),
      rng_(std::random_device{}()) {

    if (M_ == 0) {
        throw std::invalid_argument(
            "HNSW M must be greater than zero"
        );
    }

    if (ef_construction_ == 0) {
        throw std::invalid_argument(
            "efConstruction must be greater than zero"
        );
    }

    if (ef_search_ == 0) {
        throw std::invalid_argument(
            "efSearch must be greater than zero"
        );
    }
}

void HNSWIndex::set_ef_search(std::size_t ef_search) {
    if (ef_search == 0) {
        throw std::invalid_argument(
            "efSearch must be greater than zero"
        );
    }

    ef_search_ = ef_search;
}
float HNSWIndex::distance(
    const std::vector<float>& a,
    const std::vector<float>& b
) const {
    return metric_.calculate(a, b);
}
const std::unordered_map<uint64_t, HNSWNode>&
HNSWIndex::nodes() const {
    return nodes_;
}

std::size_t HNSWIndex::M() const {
    return M_;
}

std::size_t HNSWIndex::ef_construction() const {
    return ef_construction_;
}

std::size_t HNSWIndex::ef_search() const {
    return ef_search_;
}

int HNSWIndex::random_level() {

    // Temporary simple level generation.
    // We'll replace this with the proper HNSW
    // exponential level distribution next.
    std::uniform_int_distribution<int> distribution(0, 3);

    return distribution(rng_);
}
void HNSWIndex::prune_connections(
    uint64_t node_id,
    int layer
) {
    auto it = nodes_.find(node_id);

    if (it == nodes_.end()) {
        return;
    }

    HNSWNode& node = it->second;

    if (layer >= static_cast<int>(node.neighbors.size())) {
        return;
    }

    auto& neighbors = node.neighbors[layer];

    std::size_t max_neighbors =
        max_neighbors_for_layer(layer);

    if (neighbors.size() <= max_neighbors) {
        return;
    }

    const VectorRecord* node_record =
        provider_.get(node_id);

    if (node_record == nullptr) {
        return;
    }

    // Save the old neighbors.
    std::vector<uint64_t> old_neighbors =
        neighbors;

    // Build candidates using distance from node_id.
    std::vector<SearchResult> candidates;

    candidates.reserve(
        old_neighbors.size()
    );

    for (uint64_t neighbor_id : old_neighbors) {

        if (neighbor_id == node_id) {
            continue;
        }

        const VectorRecord* neighbor_record =
            provider_.get(neighbor_id);

        if (neighbor_record == nullptr) {
            continue;
        }

        float d =
            distance(
                node_record->values,
                neighbor_record->values
            );

        candidates.push_back(
            {neighbor_id, d}
        );
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const SearchResult& a,
           const SearchResult& b) {
            return a.score < b.score;
        }
    );

    // Select the neighbors we want to keep.
    std::vector<uint64_t> new_neighbors =
        select_neighbors(
            node_id,
            candidates,
            max_neighbors
        );

    // Replace node's adjacency list.
    neighbors = new_neighbors;

    // ------------------------------------------------
    // Remove reverse edges for neighbors that were
    // dropped.
    // ------------------------------------------------

    for (uint64_t old_neighbor_id : old_neighbors) {

        bool kept =
            std::find(
                new_neighbors.begin(),
                new_neighbors.end(),
                old_neighbor_id
            ) != new_neighbors.end();

        if (kept) {
            continue;
        }

        auto neighbor_it =
            nodes_.find(old_neighbor_id);

        if (neighbor_it == nodes_.end()) {
            continue;
        }

        HNSWNode& neighbor =
            neighbor_it->second;

        if (layer >=
            static_cast<int>(
                neighbor.neighbors.size()
            )) {
            continue;
        }

        auto& reverse_neighbors =
            neighbor.neighbors[layer];

        reverse_neighbors.erase(
            std::remove(
                reverse_neighbors.begin(),
                reverse_neighbors.end(),
                node_id
            ),
            reverse_neighbors.end()
        );
    }

    // ------------------------------------------------
    // Make sure every retained edge has its reverse.
    // ------------------------------------------------

    for (uint64_t neighbor_id : new_neighbors) {

        auto neighbor_it =
            nodes_.find(neighbor_id);

        if (neighbor_it == nodes_.end()) {
            continue;
        }

        HNSWNode& neighbor =
            neighbor_it->second;

        if (layer >=
            static_cast<int>(
                neighbor.neighbors.size()
            )) {
            continue;
        }

        auto& reverse_neighbors =
            neighbor.neighbors[layer];

        bool already_exists =
            std::find(
                reverse_neighbors.begin(),
                reverse_neighbors.end(),
                node_id
            ) != reverse_neighbors.end();

        if (!already_exists) {
            reverse_neighbors.push_back(node_id);
        }
    }
}

void HNSWIndex::insert(uint64_t id) {

    // Don't insert the same ID twice.
    if (nodes_.find(id) != nodes_.end()) {
        return;
    }

    const VectorRecord* record =
        provider_.get(id);

    if (record == nullptr) {
        throw std::invalid_argument(
            "Cannot insert ID that does not exist in VectorStore"
        );
    }

    // Decide how many layers this node gets.
    int level = random_level();

    HNSWNode node;

    node.id = id;

    node.neighbors.resize(level + 1);

    nodes_.emplace(id, std::move(node));

    // First node becomes the entry point.
    if (max_level_ == -1) {

        entry_point_ = id;
        max_level_ = level;

        return;
    }

    uint64_t current = entry_point_;

    /*
     * Search from the highest layer down to
     * the layer immediately above the new node.
     */
    for (int layer = max_level_;
         layer > level;
         --layer) {

        current = greedy_search_layer(
            record->values,
            current,
            layer
        );
    }

    /*
     * Now construct connections from the new
     * node's highest layer down to layer 0.
     */
    int upper_layer =
        std::min(level, max_level_);

    for (int layer = upper_layer;
         layer >= 0;
         --layer) {

        /*
         * Find candidate neighbors around the
         * new node at this layer.
         */
        std::vector<SearchResult> candidates =
            search_layer(
                record->values,
                current,
                ef_construction_,
                layer
            );

        /*
         * Select the closest M neighbors.
         */
        std::size_t max_neighbors =
    max_neighbors_for_layer(layer);

std::vector<uint64_t> neighbors =
    select_neighbors(
        id,
        candidates,
        max_neighbors
    );

// Initially connect the new node to its selected neighbors.
nodes_.at(id).neighbors[layer] = neighbors;

// Add the reverse edges.
for (uint64_t neighbor_id : neighbors) {

    auto neighbor_it = nodes_.find(neighbor_id);

    if (neighbor_it == nodes_.end()) {
        continue;
    }

    HNSWNode& neighbor = neighbor_it->second;

    if (layer >=
        static_cast<int>(neighbor.neighbors.size())) {
        continue;
    }

    neighbor.neighbors[layer].push_back(id);
}

// Now prune the existing neighbors.
for (uint64_t neighbor_id : neighbors) {

    if (nodes_.find(neighbor_id) == nodes_.end()) {
        continue;
    }

    prune_connections(
        neighbor_id,
        layer
    );
}

// Synchronize the new node's edges with the
// actual reverse edges after pruning.
auto& final_neighbors =
    nodes_.at(id).neighbors[layer];

final_neighbors.erase(
    std::remove_if(
        final_neighbors.begin(),
        final_neighbors.end(),
        [&](uint64_t neighbor_id) {

            auto neighbor_it =
                nodes_.find(neighbor_id);

            if (neighbor_it == nodes_.end()) {
                return true;
            }

            const HNSWNode& neighbor =
                neighbor_it->second;

            if (layer >=
                static_cast<int>(
                    neighbor.neighbors.size())) {
                return true;
            }

            const auto& reverse_neighbors =
                neighbor.neighbors[layer];

            return std::find(
                reverse_neighbors.begin(),
                reverse_neighbors.end(),
                id
            ) == reverse_neighbors.end();
        }
    ),
    final_neighbors.end()
);

if (!candidates.empty()) {
    current = candidates[0].id;
}
    }

    /*
     * If the new node has a higher level than the
     * current graph, make it the new entry point.
     */
    if (level > max_level_) {

        entry_point_ = id;
        max_level_ = level;
    }
}

void HNSWIndex::remove(uint64_t id) {

    auto it = nodes_.find(id);

    if (it == nodes_.end()) {
        return;
    }

    HNSWNode& node = it->second;

    /*
     * Remove this node from every neighbor list
     * in every layer.
     */
    for (std::size_t layer = 0;
         layer < node.neighbors.size();
         ++layer) {

        for (uint64_t neighbor_id : node.neighbors[layer]) {

            auto neighbor_it =
                nodes_.find(neighbor_id);

            if (neighbor_it == nodes_.end()) {
                continue;
            }

            HNSWNode& neighbor =
                neighbor_it->second;

            if (layer >= neighbor.neighbors.size()) {
                continue;
            }

            auto& neighbors =
                neighbor.neighbors[layer];

            neighbors.erase(
                std::remove(
                    neighbors.begin(),
                    neighbors.end(),
                    id
                ),
                neighbors.end()
            );
        }
    }

    bool was_entry_point =
        (id == entry_point_);

    /*
     * Remove node itself.
     */
    nodes_.erase(it);

    /*
     * Empty graph.
     */
    if (nodes_.empty()) {
        max_level_ = -1;
        entry_point_ = 0;
        return;
    }

    /*
     * If the entry point was deleted,
     * choose another node with the highest level.
     */
    if (was_entry_point) {

        uint64_t new_entry_point = 0;
        int new_max_level = -1;

        for (const auto& [node_id, candidate] : nodes_) {

            int candidate_level =
                static_cast<int>(
                    candidate.neighbors.size()
                ) - 1;

            if (candidate_level > new_max_level) {
                new_max_level = candidate_level;
                new_entry_point = node_id;
            }
        }

        entry_point_ = new_entry_point;
        max_level_ = new_max_level;
    }
}

uint64_t HNSWIndex::entry_point() const {
    return entry_point_;
}
void HNSWIndex::print_stats() const {
    std::size_t total_edges = 0;

    std::vector<std::size_t> layer_edges(
        max_level_ + 1,
        0
    );

    for (const auto& [id, node] : nodes_) {

        for (std::size_t layer = 0;
             layer < node.neighbors.size();
             ++layer) {

            layer_edges[layer] +=
                node.neighbors[layer].size();

            total_edges +=
                node.neighbors[layer].size();
        }
    }

    std::cout << "\nHNSW Stats\n";
    std::cout << "Nodes: "
              << nodes_.size()
              << '\n';

    std::cout << "Max level: "
              << max_level_
              << '\n';

    std::cout << "Entry point: "
              << entry_point_
              << '\n';

    std::cout << "Total edges: "
              << total_edges
              << '\n';

    for (std::size_t layer = 0;
         layer < layer_edges.size();
         ++layer) {

        double average = 0.0;

        if (!nodes_.empty()) {
            average =
                static_cast<double>(
                    layer_edges[layer]
                )
                / static_cast<double>(
                    nodes_.size()
                );
        }

        std::cout
            << "Layer "
            << layer
            << " edges: "
            << layer_edges[layer]
            << " (avg "
            << average
            << ")\n";
    }
}


bool HNSWIndex::validate_graph() const {

    // ============================================
    // Check every node
    // ============================================

    for (const auto& [node_id, node] : nodes_) {

        // ----------------------------------------
        // Node ID must match map key
        // ----------------------------------------

        if (node.id != node_id) {

            std::cout
                << "GRAPH ERROR: Node ID mismatch. "
                << "Map key = " << node_id
                << ", node.id = " << node.id
                << '\n';

            return false;
        }


        // ----------------------------------------
        // Check every layer
        // ----------------------------------------

        for (std::size_t layer = 0;
             layer < node.neighbors.size();
             ++layer) {

            const auto& neighbors =
                node.neighbors[layer];


            // ------------------------------------
            // Check neighbor count
            // ------------------------------------

            std::size_t max_neighbors =
                max_neighbors_for_layer(
                    static_cast<int>(layer)
                );


            if (neighbors.size() > max_neighbors) {

                std::cout
                    << "GRAPH ERROR: Too many neighbors.\n"
                    << "Node: " << node_id << '\n'
                    << "Layer: " << layer << '\n'
                    << "Neighbors: " << neighbors.size() << '\n'
                    << "Maximum: " << max_neighbors << '\n';

                return false;
            }


            // ------------------------------------
            // Check every neighbor
            // ------------------------------------

            for (uint64_t neighbor_id : neighbors) {

                // Self-loop
                if (neighbor_id == node_id) {

                    std::cout
                        << "GRAPH ERROR: Self-loop.\n"
                        << "Node: " << node_id << '\n'
                        << "Layer: " << layer << '\n';

                    return false;
                }


                // Neighbor must exist
                auto neighbor_it =
                    nodes_.find(neighbor_id);


                if (neighbor_it == nodes_.end()) {

                    std::cout
                        << "GRAPH ERROR: Dangling edge.\n"
                        << "Node: " << node_id << '\n'
                        << "Layer: " << layer << '\n'
                        << "Neighbor: " << neighbor_id
                        << " does not exist.\n";

                    return false;
                }


                const HNSWNode& neighbor =
                    neighbor_it->second;


                // Neighbor must have this layer
                if (layer >=
                    neighbor.neighbors.size()) {

                    std::cout
                        << "GRAPH ERROR: Neighbor does not "
                           "have corresponding layer.\n"
                        << "Node: " << node_id << '\n'
                        << "Layer: " << layer << '\n'
                        << "Neighbor: " << neighbor_id << '\n';

                    return false;
                }


                // --------------------------------
                // Check reverse edge
                // --------------------------------

                const auto& reverse_neighbors =
                    neighbor.neighbors[layer];


                bool reverse_found =
                    std::find(
                        reverse_neighbors.begin(),
                        reverse_neighbors.end(),
                        node_id
                    ) != reverse_neighbors.end();


                if (!reverse_found) {

                    std::cout
                        << "GRAPH ERROR: Missing reverse edge.\n"
                        << "Node: " << node_id << '\n'
                        << "Layer: " << layer << '\n'
                        << "Neighbor: " << neighbor_id
                        << '\n';

                    return false;
                }
            }
        }
    }


    // ============================================
    // Check empty graph
    // ============================================

    if (nodes_.empty()) {

        if (max_level_ != -1) {

            std::cout
                << "GRAPH ERROR: Empty graph has "
                   "invalid max level: "
                << max_level_
                << '\n';

            return false;
        }

        return true;
    }


    // ============================================
    // Check entry point exists
    // ============================================

    if (nodes_.find(entry_point_)
        == nodes_.end()) {

        std::cout
            << "GRAPH ERROR: Entry point "
               "does not exist: "
            << entry_point_
            << '\n';

        return false;
    }


    // ============================================
    // Check actual maximum level
    // ============================================

    int actual_max_level = -1;


    for (const auto& [id, node] : nodes_) {

        int level =
            static_cast<int>(
                node.neighbors.size()
            ) - 1;


        actual_max_level =
            std::max(
                actual_max_level,
                level
            );
    }


    if (actual_max_level != max_level_) {

        std::cout
            << "GRAPH ERROR: max_level mismatch.\n"
            << "Stored max_level: "
            << max_level_
            << '\n'
            << "Actual max_level: "
            << actual_max_level
            << '\n';

        return false;
    }


    // ============================================
    // Entry point must have maximum level
    // ============================================

    const HNSWNode& entry =
        nodes_.at(entry_point_);


    int entry_level =
        static_cast<int>(
            entry.neighbors.size()
        ) - 1;


    if (entry_level != max_level_) {

        std::cout
            << "GRAPH ERROR: Entry point does not "
               "have maximum level.\n"
            << "Entry point: "
            << entry_point_
            << '\n'
            << "Entry level: "
            << entry_level
            << '\n'
            << "Max level: "
            << max_level_
            << '\n';

        return false;
    }


    return true;
}
int HNSWIndex::max_level() const {
    return max_level_;
}
void HNSWIndex::load_state(
    std::unordered_map<uint64_t, HNSWNode> nodes,
    uint64_t entry_point,
    int max_level
) {
    if (nodes.empty()) {

        if (max_level != -1) {
            throw std::invalid_argument(
                "Empty HNSW state must have max_level = -1"
            );
        }

        nodes_.clear();
        entry_point_ = 0;
        max_level_ = -1;

        return;
    }

    if (nodes.find(entry_point) == nodes.end()) {
        throw std::invalid_argument(
            "HNSW state entry point does not exist"
        );
    }

    int actual_max_level = -1;

    for (const auto& [id, node] : nodes) {

        if (node.id != id) {
            throw std::invalid_argument(
                "HNSW state contains mismatched node ID"
            );
        }

        int node_level =
            static_cast<int>(
                node.neighbors.size()
            ) - 1;

        actual_max_level =
            std::max(
                actual_max_level,
                node_level
            );
    }

    if (actual_max_level != max_level) {
        throw std::invalid_argument(
            "HNSW state max level is inconsistent"
        );
    }

    // Validate the graph before accepting it.
    for (const auto& [node_id, node] : nodes) {

        for (std::size_t layer = 0;
             layer < node.neighbors.size();
             ++layer) {

            std::size_t max_neighbors =
                max_neighbors_for_layer(
                    static_cast<int>(layer)
                );

            if (node.neighbors[layer].size()
                > max_neighbors) {

                throw std::invalid_argument(
                    "HNSW state contains too many neighbors"
                );
            }

            for (uint64_t neighbor_id :
                 node.neighbors[layer]) {

                if (neighbor_id == node_id) {
                    throw std::invalid_argument(
                        "HNSW state contains self-loop"
                    );
                }

                auto neighbor_it =
                    nodes.find(neighbor_id);

                if (neighbor_it == nodes.end()) {
                    throw std::invalid_argument(
                        "HNSW state contains dangling edge"
                    );
                }

                const HNSWNode& neighbor =
                    neighbor_it->second;

                if (layer >=
                    neighbor.neighbors.size()) {

                    throw std::invalid_argument(
                        "HNSW state has invalid neighbor layer"
                    );
                }

                const auto& reverse_neighbors =
                    neighbor.neighbors[layer];

                bool reverse_found =
                    std::find(
                        reverse_neighbors.begin(),
                        reverse_neighbors.end(),
                        node_id
                    ) != reverse_neighbors.end();

                if (!reverse_found) {
                    throw std::invalid_argument(
                        "HNSW state contains missing reverse edge"
                    );
                }
            }
        }
    }

    // Everything is valid.
    nodes_ = std::move(nodes);
    entry_point_ = entry_point;
    max_level_ = max_level;
}
std::vector<SearchResult> HNSWIndex::search(
    const std::vector<float>& query,
    std::size_t k
) const {
    if (k == 0 || nodes_.empty()) {
        return {};
    }

    if (query.empty()) {
        throw std::invalid_argument(
            "Query vector cannot be empty"
        );
    }

    uint64_t current = entry_point_;

    for (int layer = max_level_; layer > 0; --layer) {

        current = greedy_search_layer(
            query,
            current,
            layer
        );
    }

    std::vector<SearchResult> results =
        search_layer(
            query,
            current,
            std::max(k, ef_search_),
            0
        );

    if (results.size() > k) {
        results.resize(k);
    }

    return results;
}

uint64_t HNSWIndex::greedy_search_layer(
    const std::vector<float>& query,
    uint64_t entry_point,
    int layer
) const {

    auto entry_it = nodes_.find(entry_point);

    if (entry_it == nodes_.end()) {
        throw std::runtime_error(
            "Entry point does not exist in HNSW graph"
        );
    }

    const VectorRecord* entry_record =
        provider_.get(entry_point);

    if (entry_record == nullptr) {
        throw std::runtime_error(
            "Entry point does not exist in VectorStore"
        );
    }

    uint64_t current = entry_point;

    float current_distance =
        distance(query, entry_record->values);

    bool improved = true;

    while (improved) {

        improved = false;

        const HNSWNode& node =
            nodes_.at(current);

        // This node doesn't exist at this layer.
        if (layer >= static_cast<int>(node.neighbors.size())) {
            break;
        }

        for (uint64_t neighbor_id : node.neighbors[layer]) {

            const VectorRecord* neighbor_record =
                provider_.get(neighbor_id);

            if (neighbor_record == nullptr) {
                continue;
            }

            float neighbor_distance =
                distance(query, neighbor_record->values);

            if (neighbor_distance < current_distance) {

                current = neighbor_id;

                current_distance =
                    neighbor_distance;

                improved = true;

                break;
            }
        }
    }

    return current;
}

std::vector<SearchResult> HNSWIndex::search_layer(
    const std::vector<float>& query,
    uint64_t entry_point,
    std::size_t ef,
    int layer
) const {

    if (ef == 0) {
        return {};
    }

    auto entry_it = nodes_.find(entry_point);

    if (entry_it == nodes_.end()) {
        throw std::runtime_error(
            "Entry point does not exist in HNSW graph"
        );
    }

    const VectorRecord* entry_record =
        provider_.get(entry_point);

    if (entry_record == nullptr) {
        throw std::runtime_error(
            "Entry point does not exist in VectorStore"
        );
    }

    /*
     * Candidates that we still need to explore.
     *
     * Smallest distance comes first.
     */
    using Candidate =
        std::pair<float, uint64_t>;

    std::priority_queue<
        Candidate,
        std::vector<Candidate>,
        std::greater<Candidate>
    > candidates;


    /*
     * Best nodes found so far.
     *
     * Largest distance comes first.
     * This lets us quickly remove the worst
     * node when we exceed ef.
     */
    std::priority_queue<
        Candidate
    > best;


    std::unordered_set<uint64_t> visited;

    float entry_distance =
        distance(query, entry_record->values);

    candidates.push({
        entry_distance,
        entry_point
    });

    best.push({
        entry_distance,
        entry_point
    });

    visited.insert(entry_point);


    while (!candidates.empty()) {

        auto [current_distance, current_id] =
            candidates.top();

        candidates.pop();


        /*
         * If the closest candidate waiting to be
         * explored is already worse than our worst
         * result, we can stop.
         */
        if (best.size() >= ef) {

            float worst_distance =
                best.top().first;

            if (current_distance > worst_distance) {
                break;
            }
        }


        const HNSWNode& current_node =
            nodes_.at(current_id);

        if (layer >=
            static_cast<int>(current_node.neighbors.size())) {
            continue;
        }


        for (uint64_t neighbor_id :
             current_node.neighbors[layer]) {

            if (visited.find(neighbor_id) !=
                visited.end()) {
                continue;
            }

            visited.insert(neighbor_id);


            const VectorRecord* neighbor_record =
                provider_.get(neighbor_id);

            if (neighbor_record == nullptr) {
                continue;
            }


            float neighbor_distance =
                distance(
                    query,
                    neighbor_record->values
                );


            /*
             * Add this node if we still have room
             * or if it is better than our current
             * worst result.
             */
            if (best.size() < ef ||
                neighbor_distance < best.top().first) {

                candidates.push({
                    neighbor_distance,
                    neighbor_id
                });

                best.push({
                    neighbor_distance,
                    neighbor_id
                });


                if (best.size() > ef) {
                    best.pop();
                }
            }
        }
    }


    /*
     * Convert the best candidates into a normal
     * vector and sort from closest to farthest.

     */
    std::vector<SearchResult> results;

    results.reserve(best.size());

    while (!best.empty()) {

        auto [distance_value, id] =
            best.top();

        best.pop();

        results.push_back({
            id,
            distance_value
        });
    }


    std::sort(
        results.begin(),
        results.end(),
        [](const SearchResult& a,
           const SearchResult& b) {

            return a.score < b.score;
        }
    );

    return results;
}


std::size_t HNSWIndex::max_neighbors_for_layer(
    int layer
) const {
    if (layer == 0) {
        return 2 * M_;
    }

    return M_;
}
std::vector<uint64_t> HNSWIndex::select_neighbors(
    uint64_t node_id,
    const std::vector<SearchResult>& candidates,
    std::size_t max_neighbors
) const {

    std::vector<uint64_t> selected;

    if (max_neighbors == 0) {
        return selected;
    }

    const VectorRecord* node_record =
        provider_.get(node_id);

    if (node_record == nullptr) {
        return selected;
    }

    /*
     * Candidates must be considered from closest
     * to farthest from node_id.
     */
    for (const SearchResult& candidate : candidates) {

        if (candidate.id == node_id) {
            continue;
        }

        if (selected.size() >= max_neighbors) {
            break;
        }

        const VectorRecord* candidate_record =
            provider_.get(candidate.id);

        if (candidate_record == nullptr) {
            continue;
        }

        bool accept = true;

        /*
         * Check whether this candidate is redundant.
         */
        for (uint64_t selected_id : selected) {

            const VectorRecord* selected_record =
                provider_.get(selected_id);

            if (selected_record == nullptr) {
                continue;
            }

            float candidate_to_selected =
                distance(
                    candidate_record->values,
                    selected_record->values
                );

            /*
             * candidate.score is:
             *
             * distance(node_id, candidate)
             */
            float candidate_to_node =
                candidate.score;

            if (candidate_to_selected <
                candidate_to_node) {

                accept = false;
                break;
            }
        }

        if (accept) {
            selected.push_back(candidate.id);
        }
    }

    return selected;
}




}