#include "minivec/persistence.h"

#include "minivec/collection.h"
#include "minivec/hnsw_index.h"
#include "minivec/cosine_metric.h"

#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace minivec {

namespace {

// ============================================================
// File format
// ============================================================

constexpr char MAGIC[8] = {
    'M', 'V', 'E', 'C', 'I', 'D', 'X', '\0'
};

// Version 2 adds metadata to VectorRecord.
constexpr uint32_t VERSION = 3;


// ============================================================
// Safety limits
// ============================================================

constexpr uint64_t MAX_DIMENSION = 100000;

constexpr uint64_t MAX_VECTOR_COUNT = 100000000;

constexpr uint64_t MAX_LAYER_COUNT = 100;

constexpr uint64_t MAX_NEIGHBOR_COUNT = 100000;

constexpr uint64_t MAX_METADATA_FIELDS = 1000;

constexpr uint64_t MAX_STRING_LENGTH = 1000000;


// ============================================================
// Write primitive
// ============================================================
template <typename T>
void write_persist_value(
    std::ofstream& out,
    const T& value
) {
    out.write(
        reinterpret_cast<const char*>(&value),
        sizeof(T)
    );

    if (!out) {
        throw std::runtime_error(
            "Failed to write MiniVec data"
        );
    }
}


// ============================================================
// Read primitive
// ============================================================

template <typename T>
void read_persist_value(
    std::ifstream& in,
    T& value
) {
    in.read(
        reinterpret_cast<char*>(&value),
        sizeof(T)
    );

    if (!in) {
        throw std::runtime_error(
            "Unexpected end of MiniVec file"
        );
    }
}

// ============================================================
// Write vector
// ============================================================

void write_vector(
    std::ofstream& out,
    const std::vector<float>& values
) {
    uint64_t dimension =
        static_cast<uint64_t>(
            values.size()
        );

    write_persist_value(
        out,
        dimension
    );

    if (!values.empty()) {

        out.write(
            reinterpret_cast<const char*>(
                values.data()
            ),
            static_cast<std::streamsize>(
                values.size() * sizeof(float)
            )
        );

        if (!out) {
            throw std::runtime_error(
                "Failed to write vector values"
            );
        }
    }
}


// ============================================================
// Read vector
// ============================================================

std::vector<float> read_vector(
    std::ifstream& in
) {
    uint64_t dimension;

    read_persist_value(
        in,
        dimension
    );


    // --------------------------------------------------------
    // Validate dimension
    // --------------------------------------------------------

    if (dimension == 0) {
        throw std::runtime_error(
            "Invalid vector dimension: zero"
        );
    }

    if (dimension > MAX_DIMENSION) {
        throw std::runtime_error(
            "Vector dimension is too large"
        );
    }

    if (
        dimension >
        static_cast<uint64_t>(
            std::numeric_limits<std::size_t>::max()
        )
    ) {
        throw std::runtime_error(
            "Vector dimension exceeds platform limit"
        );
    }


    // --------------------------------------------------------
    // Allocate vector
    // --------------------------------------------------------

    std::vector<float> values(
        static_cast<std::size_t>(
            dimension
        )
    );


    // --------------------------------------------------------
    // Read values
    // --------------------------------------------------------

    if (!values.empty()) {

        in.read(
            reinterpret_cast<char*>(
                values.data()
            ),
            static_cast<std::streamsize>(
                values.size() * sizeof(float)
            )
        );

        if (!in) {
            throw std::runtime_error(
                "Unexpected end of MiniVec file "
                "while reading vector"
            );
        }
    }


    return values;
}


// ============================================================
// Write string
// ============================================================

void write_string(
    std::ofstream& out,
    const std::string& value
) {
    uint64_t length =
        static_cast<uint64_t>(
            value.size()
        );

    write_persist_value(
        out,
        length
    );

    if (!value.empty()) {

        out.write(
            value.data(),
            static_cast<std::streamsize>(
                value.size()
            )
        );

        if (!out) {
            throw std::runtime_error(
                "Failed to write string"
            );
        }
    }
}


// ============================================================
// Read string
// ============================================================

std::string read_string(
    std::ifstream& in
) {
    uint64_t length;

    read_persist_value(
        in,
        length
    );

    if (length > MAX_STRING_LENGTH) {
        throw std::runtime_error(
            "String is too large"
        );
    }

    if (
        length >
        static_cast<uint64_t>(
            std::numeric_limits<std::size_t>::max()
        )
    ) {
        throw std::runtime_error(
            "String length exceeds platform limit"
        );
    }

    std::string value(
        static_cast<std::size_t>(
            length
        ),
        '\0'
    );

    if (!value.empty()) {

        in.read(
            value.data(),
            static_cast<std::streamsize>(
                length
            )
        );

        if (!in) {
            throw std::runtime_error(
                "Unexpected end of MiniVec file "
                "while reading string"
            );
        }
    }

    return value;
}


// ============================================================
// Write metadata
// ============================================================

void write_metadata(
    std::ofstream& out,
    const Metadata& metadata
) {
    uint64_t metadata_count =
        static_cast<uint64_t>(
            metadata.size()
        );

    if (metadata_count > MAX_METADATA_FIELDS) {
        throw std::runtime_error(
            "Metadata field count is too large"
        );
    }

    write_persist_value(
        out,
        metadata_count
    );


    for (const auto& [key, value] :
         metadata) {

        // ----------------------------------------------------
        // Key
        // ----------------------------------------------------

        write_string(
            out,
            key
        );


        // ----------------------------------------------------
        // Type
        // ----------------------------------------------------
        //
        // 0 = string
        // 1 = double
        // 2 = bool
        //
        // ----------------------------------------------------

        uint8_t type;

        if (
            std::holds_alternative<std::string>(
                value
            )
        ) {
            type = 0;
        }
        else if (
            std::holds_alternative<double>(
                value
            )
        ) {
            type = 1;
        }
        else if (
            std::holds_alternative<bool>(
                value
            )
        ) {
            type = 2;
        }
        else {
            throw std::runtime_error(
                "Unsupported metadata type"
            );
        }

        write_persist_value(
            out,
            type
        );


        // ----------------------------------------------------
        // Value
        // ----------------------------------------------------

        switch (type) {

            case 0: {

                write_string(
                    out,
                    std::get<std::string>(
                        value
                    )
                );

                break;
            }


            case 1: {

                write_persist_value(
                    out,
                    std::get<double>(
                        value
                    )
                );

                break;
            }


            case 2: {

                uint8_t bool_value =
                    std::get<bool>(value)
                        ? 1
                        : 0;

                write_persist_value(
                    out,
                    bool_value
                );

                break;
            }


            default:

                throw std::runtime_error(
                    "Unsupported metadata type"
                );
        }
    }
}


// ============================================================
// Read metadata
// ============================================================

Metadata read_metadata(
    std::ifstream& in
) {
    Metadata metadata;

    uint64_t metadata_count;

    read_persist_value(
        in,
        metadata_count
    );

    if (
        metadata_count >
        MAX_METADATA_FIELDS
    ) {
        throw std::runtime_error(
            "Metadata field count is too large"
        );
    }


    for (
        uint64_t i = 0;
        i < metadata_count;
        ++i
    ) {

        // ----------------------------------------------------
        // Key
        // ----------------------------------------------------

        std::string key =
            read_string(in);


        // ----------------------------------------------------
        // Type
        // ----------------------------------------------------

        uint8_t type;

        read_persist_value(
            in,
            type
        );


        // ----------------------------------------------------
        // Value
        // ----------------------------------------------------

        switch (type) {

            case 0: {

                metadata[key] =
                    read_string(in);

                break;
            }


            case 1: {

                double value;

                read_persist_value(
                    in,
                    value
                );

                metadata[key] =
                    value;

                break;
            }


            case 2: {

                uint8_t bool_value;

                read_persist_value(
                    in,
                    bool_value
                );

                if (bool_value > 1) {
                    throw std::runtime_error(
                        "Invalid boolean metadata value"
                    );
                }

                metadata[key] =
                    (bool_value == 1);

                break;
            }


            default:

                throw std::runtime_error(
                    "Unknown metadata type"
                );
        }
    }

    return metadata;
}

} // anonymous namespace


// ============================================================
// SAVE
// ============================================================

void Persistence::save(
    const Collection& collection,
    const std::string& path
) {

    // ========================================================
    // Get HNSW index
    // ========================================================

    const auto* hnsw =
        dynamic_cast<const HNSWIndex*>(
            collection.index_.get()
        );

    if (hnsw == nullptr) {
        throw std::runtime_error(
            "Persistence currently supports HNSWIndex only"
        );
    }


    // ========================================================
    // Temporary file
    // ========================================================

    const std::string temp_path =
        path + ".tmp";


    // Remove stale temporary file if present.

    {
        std::error_code ec;

        std::filesystem::remove(
            temp_path,
            ec
        );
    }


    // ========================================================
    // Open temporary file
    // ========================================================

    std::ofstream out(
        temp_path,
        std::ios::binary |
        std::ios::trunc
    );

    if (!out) {
        throw std::runtime_error(
            "Failed to open temporary persistence file"
        );
    }


    try {

        // ====================================================
        // HEADER
        // ====================================================

        out.write(
            MAGIC,
            sizeof(MAGIC)
        );

        if (!out) {
            throw std::runtime_error(
                "Failed to write MiniVec magic"
            );
        }


        // ----------------------------------------------------
        // Version
        // ----------------------------------------------------

        write_persist_value(
            out,
            VERSION
        );


        // ----------------------------------------------------
        // Collection information
        // ----------------------------------------------------

        uint64_t dimension =
            static_cast<uint64_t>(
                collection.dimension_
            );

        uint64_t vector_count =
            static_cast<uint64_t>(
                collection.store_->size()
            );


        write_persist_value(
            out,
            dimension
        );

        write_persist_value(
            out,
            vector_count
        );


        // ====================================================
        // HNSW CONFIGURATION
        // ====================================================

        uint64_t M =
            static_cast<uint64_t>(
                hnsw->M()
            );

        uint64_t ef_construction =
            static_cast<uint64_t>(
                hnsw->ef_construction()
            );

        uint64_t ef_search =
            static_cast<uint64_t>(
                hnsw->ef_search()
            );

        uint64_t entry_point =
            hnsw->entry_point();

        int32_t max_level =
            static_cast<int32_t>(
                hnsw->max_level()
            );


        write_persist_value(
            out,
            M
        );

        write_persist_value(
            out,
            ef_construction
        );

        write_persist_value(
            out,
            ef_search
        );

        write_persist_value(
            out,
            entry_point
        );

        write_persist_value(
            out,
            max_level
        );


        // ====================================================
        // VECTOR RECORDS
        // ====================================================

        for (const auto& [id, record] :
             collection.store_->records()) {

            write_persist_value(out, id);

write_vector(out, record.values);

write_string(out, record.text);

write_metadata(out, record.metadata);
        }


        // ====================================================
        // HNSW GRAPH
        // ====================================================

        uint64_t node_count =
            static_cast<uint64_t>(
                hnsw->nodes().size()
            );

        write_persist_value(
            out,
            node_count
        );


        for (const auto& [id, node] :
             hnsw->nodes()) {

            // ------------------------------------------------
            // Node ID
            // ------------------------------------------------

            write_persist_value(
                out,
                id
            );


            // ------------------------------------------------
            // Number of layers
            // ------------------------------------------------

            uint64_t layer_count =
                static_cast<uint64_t>(
                    node.neighbors.size()
                );

            write_persist_value(
                out,
                layer_count
            );


            // ------------------------------------------------
            // Layers
            // ------------------------------------------------

            for (const auto& neighbors :
                 node.neighbors) {

                uint64_t neighbor_count =
                    static_cast<uint64_t>(
                        neighbors.size()
                    );

                write_persist_value(
                    out,
                    neighbor_count
                );


                for (uint64_t neighbor_id :
                     neighbors) {

                    write_persist_value(
                        out,
                        neighbor_id
                    );
                }
            }
        }


        // ====================================================
        // Flush
        // ====================================================

        out.flush();

        if (!out) {
            throw std::runtime_error(
                "Failed to flush MiniVec persistence file"
            );
        }


        // ====================================================
        // Close
        // ====================================================

        out.close();

        if (!out) {
            throw std::runtime_error(
                "Failed to close MiniVec persistence file"
            );
        }


        // ====================================================
        // Publish
        // ====================================================

        std::error_code ec;


        // Remove existing database.

        std::filesystem::remove(
            path,
            ec
        );


        // Reset error state.

        ec.clear();


        // Rename temporary file.

        std::filesystem::rename(
            temp_path,
            path,
            ec
        );

        if (ec) {
            throw std::runtime_error(
                "Failed to publish MiniVec persistence file: "
                + ec.message()
            );
        }

    }
    catch (...) {

        // ----------------------------------------------------
        // Close file if still open
        // ----------------------------------------------------

        if (out.is_open()) {
            out.close();
        }


        // ----------------------------------------------------
        // Delete incomplete temporary file
        // ----------------------------------------------------

        std::error_code ec;

        std::filesystem::remove(
            temp_path,
            ec
        );


        // ----------------------------------------------------
        // Re-throw original exception
        // ----------------------------------------------------

        throw;
    }
}


// ============================================================
// LOAD
// ============================================================

std::unique_ptr<Collection>
Persistence::load(
    const std::string& path
) {

    // ========================================================
    // Open file
    // ========================================================

    std::ifstream in(
        path,
        std::ios::binary
    );

    if (!in) {
        throw std::runtime_error(
            "Failed to open MiniVec persistence file"
        );
    }


    // ========================================================
    // MAGIC
    // ========================================================

    char magic[8];

    in.read(
        magic,
        sizeof(magic)
    );

    if (!in) {
        throw std::runtime_error(
            "Failed to read MiniVec magic"
        );
    }


    if (
        std::memcmp(
            magic,
            MAGIC,
            sizeof(MAGIC)
        ) != 0
    ) {
        throw std::runtime_error(
            "Invalid MiniVec file: bad magic"
        );
    }


    // ========================================================
    // VERSION
    // ========================================================

    uint32_t version;

    read_persist_value(
        in,
        version
    );


    if (version != VERSION) {
        throw std::runtime_error(
            "Unsupported MiniVec file version: "
            + std::to_string(version)
        );
    }


    // ========================================================
    // HEADER
    // ========================================================

    uint64_t dimension;
    uint64_t vector_count;

    uint64_t M;
    uint64_t ef_construction;
    uint64_t ef_search;

    uint64_t entry_point;
    int32_t max_level;


    read_persist_value(
        in,
        dimension
    );

    read_persist_value(
        in,
        vector_count
    );

    read_persist_value(
        in,
        M
    );

    read_persist_value(
        in,
        ef_construction
    );

    read_persist_value(
        in,
        ef_search
    );

    read_persist_value(
        in,
        entry_point
    );

    read_persist_value(
        in,
        max_level
    );


    // ========================================================
    // Validate dimension
    // ========================================================

    if (dimension == 0) {
        throw std::runtime_error(
            "Invalid collection dimension"
        );
    }

    if (dimension > MAX_DIMENSION) {
        throw std::runtime_error(
            "Collection dimension is too large"
        );
    }

    if (
        dimension >
        static_cast<uint64_t>(
            std::numeric_limits<std::size_t>::max()
        )
    ) {
        throw std::runtime_error(
            "Collection dimension exceeds platform limit"
        );
    }


    // ========================================================
    // Validate vector count
    // ========================================================

    if (vector_count > MAX_VECTOR_COUNT) {
        throw std::runtime_error(
            "Vector count is too large"
        );
    }


    // ========================================================
    // Validate HNSW configuration
    // ========================================================

    if (M < 2) {
        throw std::runtime_error(
            "Invalid HNSW M"
        );
    }

    if (
        M >
        static_cast<uint64_t>(
            std::numeric_limits<std::size_t>::max()
        )
    ) {
        throw std::runtime_error(
            "HNSW M exceeds platform limit"
        );
    }


    if (ef_construction == 0) {
        throw std::runtime_error(
            "Invalid efConstruction"
        );
    }

    if (
        ef_construction >
        static_cast<uint64_t>(
            std::numeric_limits<std::size_t>::max()
        )
    ) {
        throw std::runtime_error(
            "efConstruction exceeds platform limit"
        );
    }


    if (ef_search == 0) {
        throw std::runtime_error(
            "Invalid efSearch"
        );
    }

    if (
        ef_search >
        static_cast<uint64_t>(
            std::numeric_limits<std::size_t>::max()
        )
    ) {
        throw std::runtime_error(
            "efSearch exceeds platform limit"
        );
    }


    // ========================================================
    // Validate max level
    // ========================================================

    if (max_level < -1) {
        throw std::runtime_error(
            "Invalid HNSW max level"
        );
    }


    // ========================================================
    // VECTOR STORE
    // ========================================================

    auto store =
        std::make_unique<VectorStore>();


    for (
        uint64_t i = 0;
        i < vector_count;
        ++i
    ) {

        uint64_t id;

    read_persist_value(in, id);

std::vector<float> values =
    read_vector(in);

std::string text =
    read_string(in);

Metadata metadata =
    read_metadata(in);


        // ----------------------------------------------------
        // Vector dimension must match collection dimension.
        // ----------------------------------------------------

        if (
            values.size() !=
            static_cast<std::size_t>(
                dimension
            )
        ) {
            throw std::runtime_error(
                "Vector dimension does not match "
                "collection dimension"
            );
        }


        // ----------------------------------------------------
        // Duplicate IDs are invalid.
        // ----------------------------------------------------

        if (store->get(id) != nullptr) {

            throw std::runtime_error(
                "Duplicate vector ID in MiniVec file"
            );
        }


        store->insert(
    VectorRecord{
        id,
        std::move(values),
        std::move(text),
        std::move(metadata)
    }
);
    }


    // ========================================================
    // METRIC
    // ========================================================

    auto metric =
        std::make_unique<CosineMetric>();


    // ========================================================
    // HNSW
    // ========================================================

    auto hnsw =
        std::make_unique<HNSWIndex>(
            *store,
            *metric,
            static_cast<std::size_t>(M),
            static_cast<std::size_t>(
                ef_construction
            ),
            static_cast<std::size_t>(
                ef_search
            )
        );


    // ========================================================
    // READ HNSW NODE COUNT
    // ========================================================

    uint64_t node_count;

    read_persist_value(
        in,
        node_count
    );


    if (node_count != vector_count) {

        throw std::runtime_error(
            "Vector count and HNSW node count differ"
        );
    }


    if (node_count > MAX_VECTOR_COUNT) {

        throw std::runtime_error(
            "HNSW node count is too large"
        );
    }


    // ========================================================
    // READ HNSW NODES
    // ========================================================

    std::unordered_map<
        uint64_t,
        HNSWNode
    > nodes;


    nodes.reserve(
        static_cast<std::size_t>(
            node_count
        )
    );


    for (
        uint64_t i = 0;
        i < node_count;
        ++i
    ) {

        uint64_t id;

        read_persist_value(
            in,
            id
        );


        // ----------------------------------------------------
        // Duplicate node IDs
        // ----------------------------------------------------

        if (
            nodes.find(id)
            != nodes.end()
        ) {
            throw std::runtime_error(
                "Duplicate HNSW node ID in file"
            );
        }


        // ----------------------------------------------------
        // Layer count
        // ----------------------------------------------------

        uint64_t layer_count;

        read_persist_value(
            in,
            layer_count
        );


        if (layer_count == 0) {

            throw std::runtime_error(
                "HNSW node has zero layers"
            );
        }


        if (
            layer_count >
            MAX_LAYER_COUNT
        ) {
            throw std::runtime_error(
                "HNSW layer count is too large"
            );
        }


        HNSWNode node;

        node.id = id;

        node.neighbors.resize(
            static_cast<std::size_t>(
                layer_count
            )
        );


        // ----------------------------------------------------
        // Read each layer
        // ----------------------------------------------------

        for (
            std::size_t layer = 0;
            layer < layer_count;
            ++layer
        ) {

            uint64_t neighbor_count;

            read_persist_value(
                in,
                neighbor_count
            );


            if (
                neighbor_count >
                MAX_NEIGHBOR_COUNT
            ) {
                throw std::runtime_error(
                    "HNSW neighbor count is too large"
                );
            }


            auto& neighbors =
                node.neighbors[layer];


            neighbors.resize(
                static_cast<std::size_t>(
                    neighbor_count
                )
            );


            for (
                uint64_t& neighbor_id :
                neighbors
            ) {

                read_persist_value(
                    in,
                    neighbor_id
                );
            }
        }


        nodes.emplace(
            id,
            std::move(node)
        );
    }


    // ========================================================
    // Validate HNSW nodes have vectors
    // ========================================================

    for (
        const auto& [id, node] :
        nodes
    ) {

        if (
            store->get(id)
            == nullptr
        ) {
            throw std::runtime_error(
                "HNSW node has no corresponding vector"
            );
        }
    }


    // ========================================================
    // Restore HNSW state
    // ========================================================

    hnsw->load_state(
        std::move(nodes),
        entry_point,
        max_level
    );


    // ========================================================
    // Check for trailing data
    // ========================================================

    char extra_byte;

    in.read(
        &extra_byte,
        1
    );

    if (in.gcount() != 0) {

        throw std::runtime_error(
            "MiniVec file contains unexpected trailing data"
        );
    }


    // ========================================================
    // Build final Collection
    // ========================================================

    return std::unique_ptr<Collection>(
        new Collection(
            Collection::PersistenceTag{},
            static_cast<std::size_t>(
                dimension
            ),
            std::move(store),
            std::move(metric),
            std::move(hnsw)
        )
    );
}

} // namespace minivec