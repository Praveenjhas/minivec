#pragma once

#include <string>
#include <unordered_map>
#include <variant>

namespace minivec {

using MetadataValue = std::variant<
    std::string,
    double,
    bool
>;

using Metadata =
    std::unordered_map<std::string, MetadataValue>;

}