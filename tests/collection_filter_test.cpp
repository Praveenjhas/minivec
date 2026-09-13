#include "minivec/collection.h"
#include "minivec/cosine_metric.h"
#include "minivec/hnsw_index_factory.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>

using namespace minivec;

int main() {

    Collection collection(
        4,
        std::make_unique<CosineMetric>(),
        std::make_unique<HNSWIndexFactory>(
            16,
            200,
            100
        )
    );

    collection.insert({
        1,
        {1.0f, 0.0f, 0.0f, 0.0f},
        {
            {"brand", std::string("Lenovo")},
            {"price", 80000.0}
        }
    });

    collection.insert({
        2,
        {0.99f, 0.01f, 0.0f, 0.0f},
        {
            {"brand", std::string("Dell")},
            {"price", 70000.0}
        }
    });

    collection.insert({
        3,
        {0.98f, 0.02f, 0.0f, 0.0f},
        {
            {"brand", std::string("Lenovo")},
            {"price", 90000.0}
        }
    });

    collection.insert({
        4,
        {0.0f, 1.0f, 0.0f, 0.0f},
        {
            {"brand", std::string("HP")},
            {"price", 60000.0}
        }
    });


    // Search without filter
    auto results =
        collection.search(
            {1.0f, 0.0f, 0.0f, 0.0f},
            2
        );

    assert(results.size() == 2);


    // Filter: brand == Lenovo
    FilterCondition lenovo_filter{
        "brand",
        FilterOperator::EQUAL,
        std::string("Lenovo")
    };

    auto lenovo_results =
        collection.search(
            {1.0f, 0.0f, 0.0f, 0.0f},
            2,
            lenovo_filter
        );

    assert(lenovo_results.size() == 2);

    assert(lenovo_results[0].id == 1);
    assert(lenovo_results[1].id == 3);


    // Filter: price < 75000
    FilterCondition cheap_filter{
        "price",
        FilterOperator::LESS_THAN,
        75000.0
    };

    auto cheap_results =
        collection.search(
            {1.0f, 0.0f, 0.0f, 0.0f},
            2,
            cheap_filter
        );

    assert(cheap_results.size() == 2);


    std::cout
        << "COLLECTION FILTER TEST PASSED"
        << std::endl;

    return 0;
}