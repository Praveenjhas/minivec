#pragma once

#include "minivec/metadata.h"
#include "minivec/vector_record.h"

#include <string>

namespace minivec {

enum class FilterOperator {
    EQUAL,
    NOT_EQUAL,
    LESS_THAN,
    LESS_THAN_OR_EQUAL,
    GREATER_THAN,
    GREATER_THAN_OR_EQUAL
};

struct FilterCondition {

    std::string field;

    FilterOperator op;

    MetadataValue value;
};

bool matches_filter(
    const VectorRecord& record,
    const FilterCondition& condition
);

}