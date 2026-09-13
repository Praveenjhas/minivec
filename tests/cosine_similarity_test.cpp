#include "minivec/cosine_similarity.h"

#include <gtest/gtest.h>


TEST(CosineSimilarityTest, IdenticalVectors) {

    std::vector<float> a{1.0f, 2.0f, 3.0f};

    float result = minivec::cosine_similarity(a, a);

    EXPECT_NEAR(result, 1.0f, 1e-5f);
}


TEST(CosineSimilarityTest, PerpendicularVectors) {

    std::vector<float> a{1.0f, 0.0f};
    std::vector<float> b{0.0f, 1.0f};

    float result = minivec::cosine_similarity(a, b);

    EXPECT_NEAR(result, 0.0f, 1e-5f);
}


TEST(CosineSimilarityTest, OppositeVectors) {

    std::vector<float> a{1.0f, 0.0f};
    std::vector<float> b{-1.0f, 0.0f};

    float result = minivec::cosine_similarity(a, b);

    EXPECT_NEAR(result, -1.0f, 1e-5f);
}


TEST(CosineSimilarityTest, DifferentDimensions) {

    std::vector<float> a{1.0f, 2.0f};
    std::vector<float> b{1.0f, 2.0f, 3.0f};

    EXPECT_THROW(
        minivec::cosine_similarity(a, b),
        std::invalid_argument
    );
}


TEST(CosineSimilarityTest, ZeroVector) {

    std::vector<float> a{0.0f, 0.0f};
    std::vector<float> b{1.0f, 2.0f};

    EXPECT_THROW(
        minivec::cosine_similarity(a, b),
        std::invalid_argument
    );
}