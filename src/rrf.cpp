#include "minivec/rrf.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace minivec {

RRF::RRF(std::size_t k)
    : k_(k) {

    if (k == 0) {
        throw std::invalid_argument(
            "RRF constant must be positive"
        );
    }
}

std::vector<SearchResult> RRF::fuse(
    const std::vector<SearchResult>& vector_results,
    const std::vector<SearchResult>& keyword_results,
    std::size_t top_k
) const {

    if (top_k == 0) {
        return {};
    }

    /*
     * We only care about the rank of a document
     * in each result list.
     *
     * The original score is deliberately ignored.
     */
    std::unordered_map<uint64_t, double> scores;

    /*
     * Vector search ranking.
     *
     * HNSW returns:
     *
     * rank 1 -> best vector match
     * rank 2 -> second best
     * ...
     */
    for (std::size_t i = 0;
         i < vector_results.size();
         ++i) {

        uint64_t id =
            vector_results[i].id;

        std::size_t rank = i + 1;

        scores[id] +=
            1.0 /
            static_cast<double>(k_ + rank);
    }

    /*
     * Keyword/BM25 ranking.
     *
     * BM25 returns:
     *
     * rank 1 -> highest BM25 score
     * rank 2 -> second highest
     * ...
     */
    for (std::size_t i = 0;
         i < keyword_results.size();
         ++i) {

        uint64_t id =
            keyword_results[i].id;

        std::size_t rank = i + 1;

        scores[id] +=
            1.0 /
            static_cast<double>(k_ + rank);
    }

    /*
     * Convert the map into SearchResult.
     *
     * For hybrid search, score means:
     *
     *     RRF fusion score
     *
     * Higher is better.
     */
    std::vector<SearchResult> results;

    results.reserve(scores.size());

    for (const auto& [id, score] : scores) {

        results.push_back(
            SearchResult{
                id,
                static_cast<float>(score)
            }
        );
    }

    /*
     * RRF score:
     *
     * higher = better
     */
    std::sort(
        results.begin(),
        results.end(),
        [](const SearchResult& a,
           const SearchResult& b) {

            if (a.score != b.score) {
                return a.score > b.score;
            }

            /*
             * Deterministic tie-breaking.
             */
            return a.id < b.id;
        }
    );

    if (results.size() > top_k) {
        results.resize(top_k);
    }

    return results;
}

}