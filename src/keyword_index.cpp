#include "minivec/keyword_index.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace minivec {


KeywordIndex::KeywordIndex(
    double k1,
    double b
)
    : k1_(k1),
      b_(b) {

    if (k1 < 0.0) {
        throw std::invalid_argument(
            "k1 must be non-negative"
        );
    }

    if (b < 0.0 || b > 1.0) {
        throw std::invalid_argument(
            "b must be between 0 and 1"
        );
    }
}


std::vector<std::string>
KeywordIndex::tokenize(
    const std::string& text
) {
    std::vector<std::string> tokens;

    std::string current;

    for (unsigned char c : text) {

        if (std::isalnum(c)) {

            current.push_back(
                static_cast<char>(
                    std::tolower(c)
                )
            );

        } else {

            if (!current.empty()) {

                tokens.push_back(current);

                current.clear();
            }
        }
    }

    if (!current.empty()) {
        tokens.push_back(current);
    }

    return tokens;
}


void KeywordIndex::insert(
    uint64_t id,
    const std::string& text
) {
    auto tokens = tokenize(text);

    /*
     * Count how many times each term occurs
     * in this document.
     *
     * Example:
     *
     * "laptop laptop lenovo"
     *
     * becomes:
     *
     * laptop -> 2
     * lenovo -> 1
     */
    std::unordered_map<
        std::string,
        std::size_t
    > term_frequencies;

    for (const auto& token : tokens) {
        ++term_frequencies[token];
    }

    /*
     * Add this document to the inverted index.
     *
     * term -> document ID -> frequency
     */
    for (const auto& [term, frequency] :
         term_frequencies) {

        inverted_index_[term][id] = frequency;
    }

    /*
     * Store document length.
     */
    document_lengths_[id] = tokens.size();

    ++total_documents_;

    total_terms_ += tokens.size();
}


void KeywordIndex::remove(uint64_t id) {

    auto length_it =
        document_lengths_.find(id);

    /*
     * Document does not exist.
     */
    if (length_it == document_lengths_.end()) {
        return;
    }

    /*
     * Remove this document from every posting list.
     */
    for (auto it = inverted_index_.begin();
         it != inverted_index_.end();) {

        it->second.erase(id);

        /*
         * If no document contains this term anymore,
         * remove the term itself.
         */
        if (it->second.empty()) {

            it = inverted_index_.erase(it);

        } else {

            ++it;
        }
    }

    /*
     * Update global statistics.
     */
    total_terms_ -= length_it->second;

    document_lengths_.erase(length_it);

    --total_documents_;
}


std::size_t KeywordIndex::document_count() const {
    return total_documents_;
}


std::size_t KeywordIndex::total_terms() const {
    return total_terms_;
}


double KeywordIndex::average_document_length() const {

    if (total_documents_ == 0) {
        return 0.0;
    }

    return static_cast<double>(total_terms_) /
           static_cast<double>(total_documents_);
}


std::vector<SearchResult>
KeywordIndex::search(
    const std::string& query,
    std::size_t k
) const {

    /*
     * Nothing to search.
     */
    if (k == 0 || total_documents_ == 0) {
        return {};
    }

    /*
     * Convert query into normalized terms.
     *
     * "Lenovo Laptop"
     *
     * becomes:
     *
     * ["lenovo", "laptop"]
     */
    auto query_tokens = tokenize(query);

    if (query_tokens.empty()) {
        return {};
    }

    const double avg_dl =
        average_document_length();

    if (avg_dl == 0.0) {
        return {};
    }

    /*
     * document ID -> accumulated BM25 score
     */
    std::unordered_map<
        uint64_t,
        double
    > scores;


    /*
     * Process every term in the query.
     */
    for (const auto& term : query_tokens) {

        /*
         * Find the posting list for this term.
         */
        auto posting_it =
            inverted_index_.find(term);

        /*
         * Query term doesn't occur in any
         * document.
         */
        if (posting_it ==
            inverted_index_.end()) {

            continue;
        }

        const PostingList& postings =
            posting_it->second;


        /*
         * DF = number of documents containing
         * this term.
         */
        const double df =
            static_cast<double>(
                postings.size()
            );


        /*
         * N = total number of documents.
         */
        const double N =
            static_cast<double>(
                total_documents_
            );


        /*
         * BM25 IDF:
         *
         * IDF(t) =
         *
         * log(
         *     1 +
         *     (N - DF + 0.5)
         *     /
         *     (DF + 0.5)
         * )
         */
        const double idf =
            std::log(
                1.0 +
                (N - df + 0.5) /
                (df + 0.5)
            );


        /*
         * Score every document containing
         * this query term.
         */
        for (const auto& [doc_id, tf] :
             postings) {

            auto length_it =
                document_lengths_.find(doc_id);

            if (length_it ==
                document_lengths_.end()) {

                continue;
            }


            /*
             * |D| = document length
             */
            const double document_length =
                static_cast<double>(
                    length_it->second
                );


            /*
             * TF(t,D)
             */
            const double term_frequency =
                static_cast<double>(tf);


            /*
             * BM25 denominator:
             *
             * TF +
             * k1 * (
             *     1 - b +
             *     b * |D| / avgDL
             * )
             */
            const double denominator =
                term_frequency +
                k1_ *
                (
                    1.0 -
                    b_ +
                    b_ *
                    (
                        document_length /
                        avg_dl
                    )
                );


            /*
             * TF component:
             *
             * TF * (k1 + 1)
             * ----------------
             * denominator
             */
            const double tf_component =
                (
                    term_frequency *
                    (k1_ + 1.0)
                ) /
                denominator;


            /*
             * Add this term's contribution
             * to the document's total score.
             */
            scores[doc_id] +=
                idf * tf_component;
        }
    }


    /*
     * Convert the score map into SearchResult.
     */
    std::vector<SearchResult> results;

    results.reserve(scores.size());

    for (const auto& [doc_id, score] :
         scores) {

        results.push_back(
            {
                doc_id,
                static_cast<float>(score)
            }
        );
    }


    /*
     * BM25:
     *
     * HIGHER score = better match.
     *
     * This is different from our HNSW
     * distance where LOWER = better.
     */
    const std::size_t result_count =
        std::min(k, results.size());

    std::partial_sort(
        results.begin(),
        results.begin() + result_count,
        results.end(),
        [](const SearchResult& a,
           const SearchResult& b) {

            return a.score > b.score;
        }
    );


    /*
     * Keep only top-K.
     */
    if (results.size() > k) {
        results.resize(k);
    }

    return results;
}


}