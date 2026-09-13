#include "minivec/filter.h"

#include <cassert>
#include <iostream>
#include <string>

using namespace minivec;

int main() {

    VectorRecord record;

    record.id = 1;

    record.values = {
        1.0f,
        0.0f,
        0.0f,
        0.0f
    };

    record.metadata["brand"] =
        std::string("Lenovo");

    record.metadata["price"] =
        80000.0;

    record.metadata["available"] =
        true;


    // brand == Lenovo
    assert(
        matches_filter(
            record,
            {
                "brand",
                FilterOperator::EQUAL,
                std::string("Lenovo")
            }
        )
    );


    // brand == Dell → false
    assert(
        !matches_filter(
            record,
            {
                "brand",
                FilterOperator::EQUAL,
                std::string("Dell")
            }
        )
    );


    // price < 100000 → true
    assert(
        matches_filter(
            record,
            {
                "price",
                FilterOperator::LESS_THAN,
                100000.0
            }
        )
    );


    // price > 100000 → false
    assert(
        !matches_filter(
            record,
            {
                "price",
                FilterOperator::GREATER_THAN,
                100000.0
            }
        )
    );


    // price >= 80000 → true
    assert(
        matches_filter(
            record,
            {
                "price",
                FilterOperator::GREATER_THAN_OR_EQUAL,
                80000.0
            }
        )
    );


    // available == true
    assert(
        matches_filter(
            record,
            {
                "available",
                FilterOperator::EQUAL,
                true
            }
        )
    );


    // Missing field → false
    assert(
        !matches_filter(
            record,
            {
                "category",
                FilterOperator::EQUAL,
                std::string("laptop")
            }
        )
    );


    // Type mismatch → false
    assert(
        !matches_filter(
            record,
            {
                "price",
                FilterOperator::EQUAL,
                std::string("80000")
            }
        )
    );


    std::cout
        << "FILTER TEST PASSED"
        << std::endl;

    return 0;
}