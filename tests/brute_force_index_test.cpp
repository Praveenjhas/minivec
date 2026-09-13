#include "minivec/brute_force_index.h"

#include <gtest/gtest.h>


TEST(BruteForceIndexTest, ReturnsMostSimilarVector) {

    minivec::BruteForceIndex index;

    index.insert({
        1,
        {1.0f, 0.0f}
    });

    index.insert({
        2,
        {0.0f, 1.0f}
    });

    auto results = index.search(
        {0.9f, 0.1f},
        1
    );

    ASSERT_EQ(results.size(), 1);

    EXPECT_EQ(results[0].id, 1);
}


TEST(BruteForceIndexTest, ReturnsResultsInCorrectOrder) {

    minivec::BruteForceIndex index;

    index.insert({
        1,
        {1.0f, 0.0f}
    });

    index.insert({
        2,
        {0.0f, 1.0f}
    });

    index.insert({
        3,
        {-1.0f, 0.0f}
    });

    auto results = index.search(
        {1.0f, 0.0f},
        3
    );

    ASSERT_EQ(results.size(), 3);

    EXPECT_EQ(results[0].id, 1);
    EXPECT_EQ(results[1].id, 2);
    EXPECT_EQ(results[2].id, 3);
}


TEST(BruteForceIndexTest, ReturnsOnlyKResults) {

    minivec::BruteForceIndex index;

    index.insert({1, {1.0f, 0.0f}});
    index.insert({2, {0.0f, 1.0f}});
    index.insert({3, {-1.0f, 0.0f}});

    auto results = index.search(
        {1.0f, 0.0f},
        2
    );

    EXPECT_EQ(results.size(), 2);
}


TEST(BruteForceIndexTest, KZeroReturnsEmpty) {

    minivec::BruteForceIndex index;

    index.insert({
        1,
        {1.0f, 0.0f}
    });

    auto results = index.search(
        {1.0f, 0.0f},
        0
    );

    EXPECT_TRUE(results.empty());
}


TEST(BruteForceIndexTest, RemoveWorks) {

    minivec::BruteForceIndex index;

    index.insert({
        1,
        {1.0f, 0.0f}
    });

    index.insert({
        2,
        {0.0f, 1.0f}
    });

    index.remove(1);

    auto results = index.search(
        {1.0f, 0.0f},
        2
    );

    ASSERT_EQ(results.size(), 1);

    EXPECT_EQ(results[0].id, 2);
}