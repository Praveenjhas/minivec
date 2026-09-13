#include "minivec/server.h"
#include "minivec/metadata.h"
#include "minivec/filter.h"
#include "minivec/cosine_metric.h"
#include "minivec/hnsw_index_factory.h"
#include "minivec/persistence.h"

#include "crow.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace minivec {

// ================================================================
// Helper: Parse metadata filter
// ================================================================

bool parse_filter(
    const crow::json::rvalue& filter_json,
    FilterCondition& filter,
    std::string& error
) {

    // ------------------------------------------------------------
    // field
    // ------------------------------------------------------------

    if (!filter_json.has("field")) {
        error = "filter.field is required";
        return false;
    }

    if (
        filter_json["field"].t()
        != crow::json::type::String
    ) {
        error = "filter.field must be a string";
        return false;
    }

    filter.field =
        std::string(
            filter_json["field"].s()
        );


    // ------------------------------------------------------------
    // operator
    // ------------------------------------------------------------

    if (!filter_json.has("op")) {
        error = "filter.op is required";
        return false;
    }

    if (
        filter_json["op"].t()
        != crow::json::type::String
    ) {
        error = "filter.op must be a string";
        return false;
    }

    std::string op =
        std::string(
            filter_json["op"].s()
        );

    if (op == "EQUAL") {

        filter.op =
            FilterOperator::EQUAL;

    }
    else if (op == "NOT_EQUAL") {

        filter.op =
            FilterOperator::NOT_EQUAL;

    }
    else if (op == "LESS_THAN") {

        filter.op =
            FilterOperator::LESS_THAN;

    }
    else if (op == "LESS_THAN_OR_EQUAL") {

        filter.op =
            FilterOperator::LESS_THAN_OR_EQUAL;

    }
    else if (op == "GREATER_THAN") {

        filter.op =
            FilterOperator::GREATER_THAN;

    }
    else if (op == "GREATER_THAN_OR_EQUAL") {

        filter.op =
            FilterOperator::GREATER_THAN_OR_EQUAL;

    }
    else {

        error =
            "Unknown filter operator";

        return false;
    }


    // ------------------------------------------------------------
    // value
    // ------------------------------------------------------------

    if (!filter_json.has("value")) {

        error =
            "filter.value is required";

        return false;
    }

    const auto& value =
        filter_json["value"];


    switch (value.t()) {

        case crow::json::type::String:

            filter.value =
                std::string(
                    value.s()
                );

            break;


        case crow::json::type::Number:

            filter.value =
                value.d();

            break;


        case crow::json::type::True:

            filter.value =
                true;

            break;


        case crow::json::type::False:

            filter.value =
                false;

            break;


        default:

            error =
                "filter.value must be "
                "string, number, or boolean";

            return false;
    }


    return true;
}


// ================================================================
// Helper: Parse vector from JSON
// ================================================================

bool parse_vector(
    const crow::json::rvalue& json_vector,
    std::vector<float>& values,
    std::string& error
) {

    if (
        json_vector.t()
        != crow::json::type::List
    ) {

        error =
            "Vector must be an array";

        return false;
    }


    values.clear();

    values.reserve(
        json_vector.size()
    );


    for (
        const auto& value :
        json_vector
    ) {

        if (
            value.t()
            != crow::json::type::Number
        ) {

            error =
                "Vector must contain only numbers";

            return false;
        }


        values.push_back(
            static_cast<float>(
                value.d()
            )
        );
    }


    return true;
}


// ================================================================
// Server
// ================================================================

Server::Server()
    : data_directory_("data") {

    std::filesystem::create_directories(
        data_directory_
    );

    load_collections();
}


// ================================================================
// Save collection
// ================================================================

void Server::save_collection(
    const std::string& name,
    const Collection& collection
) {

    std::string path =
        data_directory_
        + "/"
        + name
        + ".mvdb";


    Persistence::save(
        collection,
        path
    );
}


// ================================================================
// Load collections
// ================================================================

void Server::load_collections() {

    for (
        const auto& entry :
        std::filesystem::directory_iterator(
            data_directory_
        )
    ) {

        if (!entry.is_regular_file()) {
            continue;
        }


        if (
            entry.path().extension()
            != ".mvdb"
        ) {
            continue;
        }


        std::string filename =
            entry.path()
                .filename()
                .string();


        std::string name =
            entry.path()
                .stem()
                .string();


        try {

            auto collection =
                Persistence::load(
                    entry.path().string()
                );


            collections_[name] =
                std::move(collection);


            std::cout
                << "Loaded collection: "
                << name
                << " ("
                << collections_[name]->size()
                << " vectors)"
                << std::endl;
        }
        catch (
            const std::exception& e
        ) {

            std::cerr
                << "Failed to load collection "
                << filename
                << ": "
                << e.what()
                << std::endl;
        }
    }
}


// ================================================================
// Run server
// ================================================================

void Server::run(int port) {

    crow::SimpleApp app;


    // ============================================================
    // Health
    // ============================================================

    CROW_ROUTE(
        app,
        "/health"
    )
    ([]() {

        crow::json::wvalue response;

        response["status"] =
            "ok";

        return response;
    });


    // ============================================================
    // Create collection
    // ============================================================

    CROW_ROUTE(
        app,
        "/collections/<string>"
    )
    .methods(
        crow::HTTPMethod::POST
    )
    ([this](
        const crow::request& req,
        const std::string& name
    ) {

        std::lock_guard<std::mutex> lock(
            collections_mutex_
        );


        // --------------------------------------------------------
        // Duplicate check
        // --------------------------------------------------------

        if (
            collections_.find(name)
            != collections_.end()
        ) {

            crow::json::wvalue response;

            response["error"] =
                "Collection already exists";

            return crow::response(
                409,
                response
            );
        }


        // --------------------------------------------------------
        // Parse JSON
        // --------------------------------------------------------

        auto body =
            crow::json::load(
                req.body
            );


        if (!body) {

            crow::json::wvalue response;

            response["error"] =
                "Invalid JSON";

            return crow::response(
                400,
                response
            );
        }


        // --------------------------------------------------------
        // Dimension
        // --------------------------------------------------------

        if (!body.has("dimension")) {

            crow::json::wvalue response;

            response["error"] =
                "Missing dimension";

            return crow::response(
                400,
                response
            );
        }


        int dimension =
            body["dimension"].i();


        if (dimension <= 0) {

            crow::json::wvalue response;

            response["error"] =
                "Dimension must be positive";

            return crow::response(
                400,
                response
            );
        }


        // --------------------------------------------------------
        // Metric
        // --------------------------------------------------------

        auto metric =
            std::make_unique<CosineMetric>();


        // --------------------------------------------------------
        // HNSW
        // --------------------------------------------------------

        auto factory =
            std::make_unique<HNSWIndexFactory>(
                16,     // M
                200,    // efConstruction
                50      // efSearch
            );


        // --------------------------------------------------------
        // Collection
        // --------------------------------------------------------

        auto collection =
            std::make_unique<Collection>(
                static_cast<std::size_t>(
                    dimension
                ),
                std::move(metric),
                std::move(factory)
            );


        collections_[name] =
            std::move(collection);


        // --------------------------------------------------------
        // Persist
        // --------------------------------------------------------

        save_collection(
            name,
            *collections_[name]
        );


        // --------------------------------------------------------
        // Response
        // --------------------------------------------------------

        crow::json::wvalue response;

        response["message"] =
            "Collection created";

        response["name"] =
            name;

        response["dimension"] =
            dimension;


        return crow::response(
            201,
            response
        );
    });


    // ============================================================
    // Insert / Upsert vector
    // ============================================================

    CROW_ROUTE(
        app,
        "/collections/<string>/vectors"
    )
    .methods(
        crow::HTTPMethod::POST
    )
    ([this](
        const crow::request& req,
        const std::string& name
    ) {

        std::lock_guard<std::mutex> lock(
            collections_mutex_
        );


        // --------------------------------------------------------
        // Collection
        // --------------------------------------------------------

        auto it =
            collections_.find(name);


        if (
            it == collections_.end()
        ) {

            crow::json::wvalue response;

            response["error"] =
                "Collection not found";

            return crow::response(
                404,
                response
            );
        }


        // --------------------------------------------------------
        // JSON
        // --------------------------------------------------------

        auto body =
            crow::json::load(
                req.body
            );


        if (!body) {

            crow::json::wvalue response;

            response["error"] =
                "Invalid JSON";

            return crow::response(
                400,
                response
            );
        }


        // --------------------------------------------------------
        // ID
        // --------------------------------------------------------

        if (!body.has("id")) {

            crow::json::wvalue response;

            response["error"] =
                "Missing id";

            return crow::response(
                400,
                response
            );
        }


        uint64_t id =
            body["id"].u();


        // --------------------------------------------------------
        // Vector
        // --------------------------------------------------------

        if (!body.has("vector")) {

            crow::json::wvalue response;

            response["error"] =
                "Missing vector";

            return crow::response(
                400,
                response
            );
        }


        std::vector<float> values;

        std::string parse_error;


        if (
            !parse_vector(
                body["vector"],
                values,
                parse_error
            )
        ) {

            crow::json::wvalue response;

            response["error"] =
                parse_error;

            return crow::response(
                400,
                response
            );
        }


        // --------------------------------------------------------
        // Dimension
        // --------------------------------------------------------

        if (
            values.size()
            != it->second->dimension()
        ) {

            crow::json::wvalue response;

            response["error"] =
                "Vector dimension does not "
                "match collection dimension";

            return crow::response(
                400,
                response
            );
        }


        // --------------------------------------------------------
        // Text
        // --------------------------------------------------------

        if (!body.has("text")) {

            crow::json::wvalue response;

            response["error"] =
                "Missing text";

            return crow::response(
                400,
                response
            );
        }


        if (
            body["text"].t()
            != crow::json::type::String
        ) {

            crow::json::wvalue response;

            response["error"] =
                "Text must be a string";

            return crow::response(
                400,
                response
            );
        }


        std::string text =
            std::string(
                body["text"].s()
            );


        // --------------------------------------------------------
        // Metadata
        // --------------------------------------------------------

        Metadata metadata;


        if (body.has("metadata")) {

            const auto& json_metadata =
                body["metadata"];


            if (
                json_metadata.t()
                != crow::json::type::Object
            ) {

                crow::json::wvalue response;

                response["error"] =
                    "Metadata must be an object";

                return crow::response(
                    400,
                    response
                );
            }


            for (
                const auto& key :
                json_metadata.keys()
            ) {

                const auto& value =
                    json_metadata[key];


                switch (value.t()) {

                    case crow::json::type::String:

                        metadata[key] =
                            std::string(
                                value.s()
                            );

                        break;


                    case crow::json::type::Number:

                        metadata[key] =
                            value.d();

                        break;


                    case crow::json::type::True:

                        metadata[key] =
                            true;

                        break;


                    case crow::json::type::False:

                        metadata[key] =
                            false;

                        break;


                    default: {

                        crow::json::wvalue response;

                        response["error"] =
                            "Metadata values must be "
                            "string, number, or boolean";

                        return crow::response(
                            400,
                            response
                        );
                    }
                }
            }
        }


        // --------------------------------------------------------
        // Insert
        // --------------------------------------------------------

        try {

            it->second->insert(
                VectorRecord{
                    id,
                    std::move(values),
                    std::move(text),
                    std::move(metadata)
                }
            );


            // ----------------------------------------------------
            // Persist
            // ----------------------------------------------------

            save_collection(
                name,
                *it->second
            );
        }
        catch (
            const std::exception& e
        ) {

            crow::json::wvalue response;

            response["error"] =
                e.what();

            return crow::response(
                400,
                response
            );
        }


        // --------------------------------------------------------
        // Response
        // --------------------------------------------------------

        crow::json::wvalue response;

        response["message"] =
            "Vector inserted";

        response["id"] =
            id;


        return crow::response(
            201,
            response
        );
    });


    // ============================================================
    // Get vector
    // ============================================================

    CROW_ROUTE(
        app,
        "/collections/<string>/vectors/<string>"
    )
    .methods(
        crow::HTTPMethod::GET
    )
    ([this](
        const crow::request&,
        const std::string& name,
        const std::string& id_string
    ) {

        std::lock_guard<std::mutex> lock(
            collections_mutex_
        );


        auto it =
            collections_.find(name);


        if (
            it == collections_.end()
        ) {

            crow::json::wvalue response;

            response["error"] =
                "Collection not found";

            return crow::response(
                404,
                response
            );
        }


        uint64_t id;


        try {

            id =
                std::stoull(
                    id_string
                );
        }
        catch (
            const std::exception&
        ) {

            crow::json::wvalue response;

            response["error"] =
                "Invalid vector ID";

            return crow::response(
                400,
                response
            );
        }


        auto record =
            it->second->get(id);


        if (!record.has_value()) {

            crow::json::wvalue response;

            response["error"] =
                "Vector not found";

            return crow::response(
                404,
                response
            );
        }


        crow::json::wvalue response;


        // --------------------------------------------------------
        // ID
        // --------------------------------------------------------

        response["id"] =
            record->id;


        // --------------------------------------------------------
        // Vector
        // --------------------------------------------------------

        crow::json::wvalue::list vector_json;


        for (
            float value :
            record->values
        ) {

            vector_json.emplace_back(
                value
            );
        }


        response["vector"] =
            std::move(vector_json);


        // --------------------------------------------------------
        // Text
        // --------------------------------------------------------

        response["text"] =
            record->text;


        // --------------------------------------------------------
        // Metadata
        // --------------------------------------------------------

        crow::json::wvalue metadata_json;


        for (
            const auto& [key, value] :
            record->metadata
        ) {

            std::visit(
                [&](
                    const auto& actual_value
                ) {

                    metadata_json[key] =
                        actual_value;
                },
                value
            );
        }


        response["metadata"] =
            std::move(metadata_json);


        return crow::response(
            200,
            response
        );
    });


    // ============================================================
    // Delete vector
    // ============================================================

#ifdef DELETE
#undef DELETE
#endif

    CROW_ROUTE(
        app,
        "/collections/<string>/vectors/<string>"
    )
    .methods(
        crow::HTTPMethod::DELETE
    )
    ([this](
        const crow::request&,
        const std::string& name,
        const std::string& id_string
    ) {

        std::lock_guard<std::mutex> lock(
            collections_mutex_
        );


        auto it =
            collections_.find(name);


        if (
            it == collections_.end()
        ) {

            crow::json::wvalue response;

            response["error"] =
                "Collection not found";

            return crow::response(
                404,
                response
            );
        }


        uint64_t id;


        try {

            id =
                std::stoull(
                    id_string
                );
        }
        catch (
            const std::exception&
        ) {

            crow::json::wvalue response;

            response["error"] =
                "Invalid vector ID";

            return crow::response(
                400,
                response
            );
        }


        auto record =
            it->second->get(id);


        if (!record.has_value()) {

            crow::json::wvalue response;

            response["error"] =
                "Vector not found";

            return crow::response(
                404,
                response
            );
        }


        try {

            it->second->remove(id);


            save_collection(
                name,
                *it->second
            );
        }
        catch (
            const std::exception& e
        ) {

            crow::json::wvalue response;

            response["error"] =
                e.what();

            return crow::response(
                400,
                response
            );
        }


        crow::json::wvalue response;

        response["message"] =
            "Vector deleted";

        response["id"] =
            id;


        return crow::response(
            200,
            response
        );
    });


    // ============================================================
    // Vector Search
    // ============================================================

    CROW_ROUTE(
        app,
        "/collections/<string>/search"
    )
    .methods(
        crow::HTTPMethod::POST
    )
    ([this](
        const crow::request& req,
        const std::string& name
    ) {

        std::lock_guard<std::mutex> lock(
            collections_mutex_
        );


        auto it =
            collections_.find(name);


        if (
            it == collections_.end()
        ) {

            crow::json::wvalue response;

            response["error"] =
                "Collection not found";

            return crow::response(
                404,
                response
            );
        }


        auto body =
            crow::json::load(
                req.body
            );


        if (!body) {

            crow::json::wvalue response;

            response["error"] =
                "Invalid JSON";

            return crow::response(
                400,
                response
            );
        }


        if (!body.has("vector")) {

            crow::json::wvalue response;

            response["error"] =
                "Missing vector";

            return crow::response(
                400,
                response
            );
        }


        if (!body.has("k")) {

            crow::json::wvalue response;

            response["error"] =
                "Missing k";

            return crow::response(
                400,
                response
            );
        }


        std::vector<float> query;

        std::string parse_error;


        if (
            !parse_vector(
                body["vector"],
                query,
                parse_error
            )
        ) {

            crow::json::wvalue response;

            response["error"] =
                parse_error;

            return crow::response(
                400,
                response
            );
        }


        if (
            query.size()
            != it->second->dimension()
        ) {

            crow::json::wvalue response;

            response["error"] =
                "Query dimension does not "
                "match collection dimension";

            return crow::response(
                400,
                response
            );
        }


        int k_value =
            body["k"].i();


        if (k_value <= 0) {

            crow::json::wvalue response;

            response["error"] =
                "k must be positive";

            return crow::response(
                400,
                response
            );
        }


        std::size_t k =
            static_cast<std::size_t>(
                k_value
            );


        try {

            auto results =
                it->second->search(
                    query,
                    k
                );


            crow::json::wvalue response;

            crow::json::wvalue::list result_list;


            for (
                const auto& result :
                results
            ) {

                crow::json::wvalue item;

                item["id"] =
                    result.id;

                item["score"] =
                    result.score;


                result_list.emplace_back(
                    std::move(item)
                );
            }


            response["results"] =
                std::move(result_list);


            return crow::response(
                200,
                response
            );
        }
        catch (
            const std::exception& e
        ) {

            crow::json::wvalue response;

            response["error"] =
                e.what();

            return crow::response(
                400,
                response
            );
        }
    });


    // ============================================================
    // Keyword / BM25 Search
    // ============================================================

    CROW_ROUTE(
        app,
        "/collections/<string>/keyword-search"
    )
    .methods(
        crow::HTTPMethod::POST
    )
    ([this](
        const crow::request& req,
        const std::string& name
    ) {

        std::lock_guard<std::mutex> lock(
            collections_mutex_
        );


        auto it =
            collections_.find(name);


        if (
            it == collections_.end()
        ) {

            crow::json::wvalue response;

            response["error"] =
                "Collection not found";

            return crow::response(
                404,
                response
            );
        }


        auto body =
            crow::json::load(
                req.body
            );


        if (!body) {

            crow::json::wvalue response;

            response["error"] =
                "Invalid JSON";

            return crow::response(
                400,
                response
            );
        }


        // --------------------------------------------------------
        // Query
        // --------------------------------------------------------

        if (!body.has("query")) {

            crow::json::wvalue response;

            response["error"] =
                "Missing query";

            return crow::response(
                400,
                response
            );
        }


        if (
            body["query"].t()
            != crow::json::type::String
        ) {

            crow::json::wvalue response;

            response["error"] =
                "query must be a string";

            return crow::response(
                400,
                response
            );
        }


        std::string query =
            std::string(
                body["query"].s()
            );


        // --------------------------------------------------------
        // K
        // --------------------------------------------------------

        if (!body.has("k")) {

            crow::json::wvalue response;

            response["error"] =
                "Missing k";

            return crow::response(
                400,
                response
            );
        }


        int k_value =
            body["k"].i();


        if (k_value <= 0) {

            crow::json::wvalue response;

            response["error"] =
                "k must be positive";

            return crow::response(
                400,
                response
            );
        }


        std::size_t k =
            static_cast<std::size_t>(
                k_value
            );


        try {

            auto results =
                it->second->keyword_search(
                    query,
                    k
                );


            crow::json::wvalue response;

            crow::json::wvalue::list result_list;


            for (
                const auto& result :
                results
            ) {

                crow::json::wvalue item;

                item["id"] =
                    result.id;

                item["score"] =
                    result.score;


                result_list.emplace_back(
                    std::move(item)
                );
            }


            response["results"] =
                std::move(result_list);


            return crow::response(
                200,
                response
            );
        }
        catch (
            const std::exception& e
        ) {

            crow::json::wvalue response;

            response["error"] =
                e.what();

            return crow::response(
                400,
                response
            );
        }
    });


    // ============================================================
    // Hybrid Search + RRF + Metadata Filter
    // ============================================================
      CROW_ROUTE(
    app,
    "/collections/<string>/hybrid-search"
)
.methods(
    crow::HTTPMethod::POST
)
([this](
    const crow::request& req,
    const std::string& name
) {

    std::lock_guard<std::mutex> lock(
        collections_mutex_
    );

    // --------------------------------------------------------
    // Collection
    // --------------------------------------------------------

    auto it =
        collections_.find(name);

    if (
        it == collections_.end()
    ) {
        crow::json::wvalue response;

        response["error"] =
            "Collection not found";

        return crow::response(
            404,
            response
        );
    }

    // --------------------------------------------------------
    // JSON
    // --------------------------------------------------------

    auto body =
        crow::json::load(
            req.body
        );

    if (!body) {
        crow::json::wvalue response;

        response["error"] =
            "Invalid JSON";

        return crow::response(
            400,
            response
        );
    }

    // --------------------------------------------------------
    // Vector
    // --------------------------------------------------------

    if (!body.has("vector")) {
        crow::json::wvalue response;

        response["error"] =
            "Missing vector";

        return crow::response(
            400,
            response
        );
    }

    std::vector<float> query_vector;
    std::string parse_error;

    if (
        !parse_vector(
            body["vector"],
            query_vector,
            parse_error
        )
    ) {
        crow::json::wvalue response;

        response["error"] =
            parse_error;

        return crow::response(
            400,
            response
        );
    }

    if (
        query_vector.size()
        != it->second->dimension()
    ) {
        crow::json::wvalue response;

        response["error"] =
            "Query dimension does not "
            "match collection dimension";

        return crow::response(
            400,
            response
        );
    }

    // --------------------------------------------------------
    // Text query
    // --------------------------------------------------------

    if (!body.has("query")) {
        crow::json::wvalue response;

        response["error"] =
            "Missing query";

        return crow::response(
            400,
            response
        );
    }

    if (
        body["query"].t()
        != crow::json::type::String
    ) {
        crow::json::wvalue response;

        response["error"] =
            "query must be a string";

        return crow::response(
            400,
            response
        );
    }

    std::string query_text =
        std::string(
            body["query"].s()
        );

    // --------------------------------------------------------
    // K
    // --------------------------------------------------------

    if (!body.has("k")) {
        crow::json::wvalue response;

        response["error"] =
            "Missing k";

        return crow::response(
            400,
            response
        );
    }

    int k_value =
        body["k"].i();

    if (k_value <= 0) {
        crow::json::wvalue response;

        response["error"] =
            "k must be positive";

        return crow::response(
            400,
            response
        );
    }

    std::size_t k =
        static_cast<std::size_t>(
            k_value
        );

    // --------------------------------------------------------
    // Optional filter
    // --------------------------------------------------------

    FilterCondition filter;
    FilterCondition* filter_ptr =
        nullptr;

    if (body.has("filter")) {

        const auto& filter_json =
            body["filter"];

        if (
            filter_json.t()
            != crow::json::type::Object
        ) {
            crow::json::wvalue response;

            response["error"] =
                "filter must be an object";

            return crow::response(
                400,
                response
            );
        }

        std::string filter_error;

        if (
            !parse_filter(
                filter_json,
                filter,
                filter_error
            )
        ) {
            crow::json::wvalue response;

            response["error"] =
                filter_error;

            return crow::response(
                400,
                response
            );
        }

        filter_ptr =
            &filter;
    }

    // --------------------------------------------------------
    // Hybrid search
    // --------------------------------------------------------

    try {

        auto results =
            it->second->hybrid_search(
                query_vector,
                query_text,
                k,
                filter_ptr
            );

        // ----------------------------------------------------
        // Response
        // ----------------------------------------------------

        crow::json::wvalue response;

        crow::json::wvalue::list result_list;

        for (const auto& result : results) {

    crow::json::wvalue item;

    item["id"] = result.id;
    item["score"] = result.score;

    auto record =
        it->second->get(result.id);

    if (record.has_value()) {

        item["text"] =
            record->text;

        crow::json::wvalue metadata_json;

        for (const auto& [key, value] :
             record->metadata) {

            if (const auto* v =
                    std::get_if<std::string>(&value)) {

                metadata_json[key] = *v;

            } else if (const auto* v =
                           std::get_if<double>(&value)) {

                metadata_json[key] = *v;

            } else if (const auto* v =
                           std::get_if<bool>(&value)) {

                metadata_json[key] = *v;
            }
        }

        item["metadata"] =
            std::move(metadata_json);
    }

    result_list.emplace_back(
        std::move(item)
    );
}

        response["results"] =
            std::move(result_list);

        return crow::response(
            200,
            response
        );
    }
    catch (
        const std::exception& e
    ) {

        crow::json::wvalue response;

        response["error"] =
            e.what();

        return crow::response(
            400,
            response
        );
    }
});

    // ============================================================
    // Run server
    // ============================================================

    app
        .port(port)
        .multithreaded()
        .run();
}

} // namespace minivec