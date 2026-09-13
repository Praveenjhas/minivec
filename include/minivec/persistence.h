#pragma once

#include <memory>
#include <string>

namespace minivec {

class Collection;

class Persistence {

public:

    static void save(
        const Collection& collection,
        const std::string& path
    );

    static std::unique_ptr<Collection> load(
        const std::string& path
    );
};

}