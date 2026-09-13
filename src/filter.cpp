#include "minivec/filter.h"

#include <stdexcept>

namespace minivec {

bool matches_filter(
    const VectorRecord& record,
    const FilterCondition& condition
) {
    auto it = record.metadata.find(condition.field);

    // Field does not exist
    if (it == record.metadata.end()) {
        return false;
    }

    const MetadataValue& actual = it->second;
    const MetadataValue& expected = condition.value;

    // Types must match
    if (actual.index() != expected.index()) {
        return false;
    }

    return std::visit(
        [&](const auto& actual_value) -> bool {

            using T = std::decay_t<decltype(actual_value)>;

            const T& expected_value =
                std::get<T>(expected);

            switch (condition.op) {

                case FilterOperator::EQUAL:
                    return actual_value == expected_value;

                case FilterOperator::NOT_EQUAL:
                    return actual_value != expected_value;

                case FilterOperator::LESS_THAN:
                    return actual_value < expected_value;

                case FilterOperator::LESS_THAN_OR_EQUAL:
                    return actual_value <= expected_value;

                case FilterOperator::GREATER_THAN:
                    return actual_value > expected_value;

                case FilterOperator::GREATER_THAN_OR_EQUAL:
                    return actual_value >= expected_value;
            }

            throw std::invalid_argument(
                "Unknown filter operator"
            );
        },
        actual
    );
}

}