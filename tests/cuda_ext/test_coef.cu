#include <gtest/gtest.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include "../../libs/ccc_cuda_ext/coef.cuh"

namespace py = pybind11;

// Create a test fixture to initialize Python
class CoefTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        if (!guard) {
            guard = std::make_unique<py::scoped_interpreter>();
        }
    }

    static void TearDownTestSuite()
    {
        guard.reset();
    }

private:
    // Static member definitions
    static std::unique_ptr<py::scoped_interpreter> guard;
};

// Static member definitions
std::unique_ptr<py::scoped_interpreter> CoefTest::guard;

TEST_F(CoefTest, ExampleReturnOptionalVectors)
{
    // Test all vectors included
    {
        py::object result = example_return_optional_vectors(true, true, true);
        py::tuple tuple = result.cast<py::tuple>();

        ASSERT_EQ(tuple.size(), 3);

        // Check first vector
        ASSERT_FALSE(tuple[0].is_none());
        auto vec1 = tuple[0].cast<std::vector<float>>();
        EXPECT_EQ(vec1, std::vector<float>({1.0f, 2.0f, 3.0f}));

        // Check second vector
        ASSERT_FALSE(tuple[1].is_none());
        auto vec2 = tuple[1].cast<std::vector<int>>();
        EXPECT_EQ(vec2, std::vector<int>({4, 5, 6}));

        // Check third vector
        ASSERT_FALSE(tuple[2].is_none());
        auto vec3 = tuple[2].cast<std::vector<double>>();
        EXPECT_EQ(vec3, std::vector<double>({7.0, 8.0, 9.0}));
    }

    // Test with only first vector
    {
        py::object result = example_return_optional_vectors(true, false, false);
        py::tuple tuple = result.cast<py::tuple>();

        ASSERT_EQ(tuple.size(), 3);

        // Check first vector exists
        ASSERT_FALSE(tuple[0].is_none());
        auto vec1 = tuple[0].cast<std::vector<float>>();
        EXPECT_EQ(vec1, std::vector<float>({1.0f, 2.0f, 3.0f}));

        // Check others are None
        EXPECT_TRUE(tuple[1].is_none());
        EXPECT_TRUE(tuple[2].is_none());
    }

    // Test with no vectors
    {
        py::object result = example_return_optional_vectors(false, false, false);
        py::tuple tuple = result.cast<py::tuple>();

        ASSERT_EQ(tuple.size(), 3);
        EXPECT_TRUE(tuple[0].is_none());
        EXPECT_TRUE(tuple[1].is_none());
        EXPECT_TRUE(tuple[2].is_none());
    }
}
