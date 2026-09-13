#include "minivec/keyword_index.h"

#include <cassert>
#include <iostream>

using namespace minivec;

int main() {

    KeywordIndex index;

    index.insert(
        1,
        "Lenovo ThinkPad laptop"
    );

    index.insert(
        2,
        "Dell XPS laptop"
    );

    index.insert(
        3,
        "Lenovo monitor"
    );

    assert(index.document_count() == 3);
    assert(index.total_terms() == 8);

    index.remove(2);

    assert(index.document_count() == 2);
    assert(index.total_terms() == 5);

    std::cout << "KEYWORD INDEX TEST PASSED\n";

    return 0;
}