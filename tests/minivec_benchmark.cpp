#include "minivec/vector_store.h"
#include "minivec/brute_force_index.h"
#include "minivec/hnsw_index.h"
#include "minivec/cosine_metric.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace minivec;

using Clock = std::chrono::steady_clock;


// ============================================================
// Configuration
// ============================================================

static constexpr std::size_t NUM_VECTORS = 10000;
static constexpr std::size_t DIMENSION = 384;

static constexpr std::size_t NUM_QUERIES = 500;

static constexpr std::size_t TOP_K = 10;

static constexpr std::size_t M = 16;
static constexpr std::size_t EF_CONSTRUCTION = 200;


// ============================================================
// Statistics
// ============================================================

struct BenchmarkStats {

    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;

    double average_ms = 0.0;

    double qps = 0.0;
};


// ============================================================
// Percentile calculation
// ============================================================

double percentile(
    std::vector<double> values,
    double p
) {

    if (values.empty()) {
        return 0.0;
    }

    std::sort(
        values.begin(),
        values.end()
    );

    double index =
        (p / 100.0)
        * static_cast<double>(
            values.size() - 1
        );

    std::size_t lower =
        static_cast<std::size_t>(
            std::floor(index)
        );

    std::size_t upper =
        static_cast<std::size_t>(
            std::ceil(index)
        );

    if (lower == upper) {
        return values[lower];
    }

    double weight =
        index
        - static_cast<double>(lower);

    return
        values[lower]
        * (1.0 - weight)
        +
        values[upper]
        * weight;
}


// ============================================================
// Benchmark statistics
// ============================================================

BenchmarkStats calculate_stats(
    const std::vector<double>& latencies
) {

    BenchmarkStats stats;

    if (latencies.empty()) {
        return stats;
    }

    double total = 0.0;

    for (double latency : latencies) {
        total += latency;
    }

    stats.average_ms =
        total
        / static_cast<double>(
            latencies.size()
        );

    stats.p50_ms =
        percentile(
            latencies,
            50.0
        );

    stats.p95_ms =
        percentile(
            latencies,
            95.0
        );

    stats.p99_ms =
        percentile(
            latencies,
            99.0
        );

    double total_seconds =
        total / 1000.0;

    if (total_seconds > 0.0) {

        stats.qps =
            static_cast<double>(
                latencies.size()
            )
            / total_seconds;
    }

    return stats;
}


// ============================================================
// Recall
// ============================================================

double calculate_recall(
    const std::vector<SearchResult>& approximate,
    const std::vector<SearchResult>& ground_truth,
    std::size_t k
) {

    if (ground_truth.empty()) {
        return 0.0;
    }

    std::size_t actual_k =
        std::min(
            k,
            std::min(
                approximate.size(),
                ground_truth.size()
            )
        );

    if (actual_k == 0) {
        return 0.0;
    }

    std::size_t matches = 0;

    for (std::size_t i = 0;
         i < actual_k;
         ++i) {

        uint64_t id =
            approximate[i].id;

        for (std::size_t j = 0;
             j < actual_k;
             ++j) {

            if (
                ground_truth[j].id
                == id
            ) {

                ++matches;
                break;
            }
        }
    }

    return static_cast<double>(matches)
        / static_cast<double>(actual_k);
}


// ============================================================
// Generate normalized random vector
// ============================================================

std::vector<float> random_vector(
    std::mt19937& rng
) {

    std::normal_distribution<float> distribution(
        0.0f,
        1.0f
    );

    std::vector<float> vector(
        DIMENSION
    );

    double norm = 0.0;

    for (float& value : vector) {

        value = distribution(rng);

        norm +=
            static_cast<double>(
                value
            )
            *
            static_cast<double>(
                value
            );
    }

    norm = std::sqrt(norm);

    if (norm > 0.0) {

        for (float& value : vector) {

            value =
                static_cast<float>(
                    value / norm
                );
        }
    }

    return vector;
}


// ============================================================
// Generate dataset
// ============================================================

void build_dataset(
    VectorStore& store,
    std::mt19937& rng
) {

    std::cout
        << "\nBuilding dataset...\n";

    auto start = Clock::now();

    for (
        std::size_t i = 0;
        i < NUM_VECTORS;
        ++i
    ) {

        VectorRecord record;

        record.id =
            static_cast<uint64_t>(i);

        record.values =
            random_vector(rng);

        record.text =
            "benchmark document";

        store.insert(record);
    }

    auto end = Clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    std::cout
        << "Vectors: "
        << store.size()
        << "\n";

    std::cout
        << "Dimensions: "
        << DIMENSION
        << "\n";

    std::cout
        << "Build dataset time: "
        << std::fixed
        << std::setprecision(2)
        << elapsed_ms
        << " ms\n";
}


// ============================================================
// Generate queries
// ============================================================

std::vector<std::vector<float>> generate_queries(
    std::mt19937& rng
) {

    std::vector<std::vector<float>> queries;

    queries.reserve(
        NUM_QUERIES
    );

    for (
        std::size_t i = 0;
        i < NUM_QUERIES;
        ++i
    ) {

        queries.push_back(
            random_vector(rng)
        );
    }

    return queries;
}


// ============================================================
// Build ground truth
// ============================================================

std::vector<
    std::vector<SearchResult>
>
build_ground_truth(
    const BruteForceIndex& brute_force,
    const std::vector<
        std::vector<float>
    >& queries
) {

    std::cout
        << "\nBuilding exact ground truth...\n";

    std::vector<
        std::vector<SearchResult>
    > ground_truth;

    ground_truth.reserve(
        queries.size()
    );

    auto start = Clock::now();

    for (const auto& query : queries) {

        ground_truth.push_back(
            brute_force.search(
                query,
                TOP_K
            )
        );
    }

    auto end = Clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    std::cout
        << "Ground truth generated in "
        << elapsed_ms
        << " ms\n";

    return ground_truth;
}


// ============================================================
// Benchmark HNSW
// ============================================================

void benchmark_hnsw(
    HNSWIndex& hnsw,
    const std::vector<
        std::vector<float>
    >& queries,
    const std::vector<
        std::vector<SearchResult>
    >& ground_truth,
    std::size_t ef_search
) {

    std::cout
        << "\n"
        << "============================================================\n";

    std::cout
        << "HNSW efSearch = "
        << ef_search
        << "\n";

    std::cout
        << "============================================================\n";

    hnsw.set_ef_search(
        ef_search
    );

    std::vector<double> latencies;

    latencies.reserve(
        queries.size()
    );

    double recall_1 = 0.0;
    double recall_5 = 0.0;
    double recall_10 = 0.0;

    for (
        std::size_t i = 0;
        i < queries.size();
        ++i
    ) {

        const auto& query =
            queries[i];

        auto start =
            Clock::now();

        auto results =
            hnsw.search(
                query,
                TOP_K
            );

        auto end =
            Clock::now();

        double elapsed_us =
            std::chrono::duration<double, std::micro>(
                end - start
            ).count();

        double elapsed_ms =
            elapsed_us / 1000.0;

        latencies.push_back(
            elapsed_ms
        );

        recall_1 +=
            calculate_recall(
                results,
                ground_truth[i],
                1
            );

        recall_5 +=
            calculate_recall(
                results,
                ground_truth[i],
                5
            );

        recall_10 +=
            calculate_recall(
                results,
                ground_truth[i],
                10
            );
    }

    BenchmarkStats stats =
        calculate_stats(
            latencies
        );

    recall_1 /=
        static_cast<double>(
            queries.size()
        );

    recall_5 /=
        static_cast<double>(
            queries.size()
        );

    recall_10 /=
        static_cast<double>(
            queries.size()
        );

    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "Queries:       "
        << queries.size()
        << "\n";

    std::cout
        << "Top-K:         "
        << TOP_K
        << "\n";

    std::cout
        << "Average:       "
        << stats.average_ms
        << " ms\n";

    std::cout
        << "p50:           "
        << stats.p50_ms
        << " ms\n";

    std::cout
        << "p95:           "
        << stats.p95_ms
        << " ms\n";

    std::cout
        << "p99:           "
        << stats.p99_ms
        << " ms\n";

    std::cout
        << "QPS:           "
        << stats.qps
        << "\n";

    std::cout
        << "Recall@1:      "
        << recall_1 * 100.0
        << "%\n";

    std::cout
        << "Recall@5:      "
        << recall_5 * 100.0
        << "%\n";

    std::cout
        << "Recall@10:     "
        << recall_10 * 100.0
        << "%\n";
}


// ============================================================
// Main
// ============================================================

int main() {

    std::cout
        << "\n"
        << "============================================================\n";

    std::cout
        << "                 MINIVEC HNSW BENCHMARK\n";

    std::cout
        << "============================================================\n";

    std::cout
        << "\n";

    std::cout
        << "Vectors:       "
        << NUM_VECTORS
        << "\n";

    std::cout
        << "Dimensions:    "
        << DIMENSION
        << "\n";

    std::cout
        << "Queries:       "
        << NUM_QUERIES
        << "\n";

    std::cout
        << "Top-K:         "
        << TOP_K
        << "\n";


    // ========================================================
    // Random generator
    // ========================================================

    std::mt19937 rng(
        42
    );


    // ========================================================
    // Vector store
    // ========================================================

    VectorStore store;


    // ========================================================
    // Build dataset
    // ========================================================

    build_dataset(
        store,
        rng
    );


    // ========================================================
    // Metric
    // ========================================================

    auto metric =
        std::make_shared<CosineMetric>();


    // ========================================================
    // Brute Force
    // ========================================================

    BruteForceIndex brute_force(
        store,
        *metric
    );


    // ========================================================
    // HNSW
    // ========================================================

    HNSWIndex hnsw(
        store,
        *metric,
        M,
        EF_CONSTRUCTION,
        50
    );


    // ========================================================
    // Insert IDs into indexes
    // ========================================================

    std::cout
        << "\nBuilding indexes...\n";

    auto index_start =
        Clock::now();

    for (
        std::size_t i = 0;
        i < NUM_VECTORS;
        ++i
    ) {

        uint64_t id =
            static_cast<uint64_t>(i);

        brute_force.insert(id);

        hnsw.insert(id);
    }

    auto index_end =
        Clock::now();

    double index_build_ms =
        std::chrono::duration<double, std::milli>(
            index_end - index_start
        ).count();

    std::cout
        << "HNSW + Brute Force build time: "
        << index_build_ms
        << " ms\n";


    // ========================================================
    // Generate queries
    // ========================================================

    auto queries =
        generate_queries(
            rng
        );


    // ========================================================
    // Exact ground truth
    // ========================================================

    auto ground_truth =
        build_ground_truth(
            brute_force,
            queries
        );


    // ========================================================
    // Benchmark HNSW configurations
    // ========================================================

    const std::vector<std::size_t>
        ef_values = {
            50,
            100,
            200
        };


    for (
        std::size_t ef :
        ef_values
    ) {

        benchmark_hnsw(
            hnsw,
            queries,
            ground_truth,
            ef
        );
    }


    // ========================================================
    // Final
    // ========================================================

    std::cout
        << "\n"
        << "============================================================\n";

    std::cout
        << "                    BENCHMARK COMPLETE\n";

    std::cout
        << "============================================================\n";

    return 0;
}