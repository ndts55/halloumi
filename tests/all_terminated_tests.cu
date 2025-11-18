#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "core/types.cuh"
#include "propagation/all_terminated.cuh"
#include "propagation/cuda_utils.cuh"

TEST(ReduceBoolWithAndTest, AllTrue) {
    // Setup: Create array with all true values
    const std::size_t n_elements = 128;
    const std::size_t block_size = 64;
    const std::size_t grid_size = (n_elements + block_size - 1) / block_size;
    
    HostBoolArray host_input(n_elements, true);
    HostBoolArray host_output(grid_size, false);
    
    // Copy to device
    ASSERT_EQ(host_input.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_output.copy_to_device(), cudaSuccess);
    
    DeviceBoolArray input = host_input.get();
    DeviceBoolArray output = host_output.get();
    
    // Launch kernel
    std::size_t shared_mem_size = block_size * sizeof(bool);
    reduce_bool_with_and<<<grid_size, block_size, shared_mem_size>>>(input, output);
    
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    // Copy result back
    ASSERT_EQ(host_output.copy_to_host(), cudaSuccess);
    
    // Verify: All blocks should have reduced to true
    for (std::size_t i = 0; i < grid_size; ++i) {
        EXPECT_TRUE(host_output.at(i)) << "Block " << i << " should be true";
    }
}

TEST(ReduceBoolWithAndTest, OneFalse) {
    // Setup: Create array with one false value
    const std::size_t n_elements = 128;
    const std::size_t block_size = 64;
    const std::size_t grid_size = (n_elements + block_size - 1) / block_size;
    
    std::vector<bool> input_data(n_elements, true);
    input_data[50] = false; // Set one element to false
    
    HostBoolArray host_input(std::move(input_data));
    HostBoolArray host_output(grid_size, false);
    
    // Copy to device
    ASSERT_EQ(host_input.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_output.copy_to_device(), cudaSuccess);
    
    DeviceBoolArray input = host_input.get();
    DeviceBoolArray output = host_output.get();
    
    // Launch kernel
    std::size_t shared_mem_size = block_size * sizeof(bool);
    reduce_bool_with_and<<<grid_size, block_size, shared_mem_size>>>(input, output);
    
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    // Copy result back
    ASSERT_EQ(host_output.copy_to_host(), cudaSuccess);
    
    // Verify: The block containing the false value should be false
    // Block 0 contains elements 0-63, so element 50 is in block 0
    EXPECT_FALSE(host_output.at(0)) << "Block 0 should be false (contains element 50)";
    
    // Other blocks should be true
    for (std::size_t i = 1; i < grid_size; ++i) {
        EXPECT_TRUE(host_output.at(i)) << "Block " << i << " should be true";
    }
}

TEST(ReduceBoolWithAndTest, AllFalse) {
    // Setup: Create array with all false values
    const std::size_t n_elements = 128;
    const std::size_t block_size = 64;
    const std::size_t grid_size = (n_elements + block_size - 1) / block_size;
    
    HostBoolArray host_input(n_elements, false);
    HostBoolArray host_output(grid_size, true);
    
    // Copy to device
    ASSERT_EQ(host_input.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_output.copy_to_device(), cudaSuccess);
    
    DeviceBoolArray input = host_input.get();
    DeviceBoolArray output = host_output.get();
    
    // Launch kernel
    std::size_t shared_mem_size = block_size * sizeof(bool);
    reduce_bool_with_and<<<grid_size, block_size, shared_mem_size>>>(input, output);
    
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    // Copy result back
    ASSERT_EQ(host_output.copy_to_host(), cudaSuccess);
    
    // Verify: All blocks should have reduced to false
    for (std::size_t i = 0; i < grid_size; ++i) {
        EXPECT_FALSE(host_output.at(i)) << "Block " << i << " should be false";
    }
}

TEST(ReduceBoolWithAndTest, LastElementFalse) {
    // Setup: Test boundary condition - last element is false
    const std::size_t n_elements = 128;
    const std::size_t block_size = 64;
    const std::size_t grid_size = (n_elements + block_size - 1) / block_size;
    
    std::vector<bool> input_data(n_elements, true);
    input_data[n_elements - 1] = false; // Last element false
    
    HostBoolArray host_input(std::move(input_data));
    HostBoolArray host_output(grid_size, false);
    
    // Copy to device
    ASSERT_EQ(host_input.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_output.copy_to_device(), cudaSuccess);
    
    DeviceBoolArray input = host_input.get();
    DeviceBoolArray output = host_output.get();
    
    // Launch kernel
    std::size_t shared_mem_size = block_size * sizeof(bool);
    reduce_bool_with_and<<<grid_size, block_size, shared_mem_size>>>(input, output);
    
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    // Copy result back
    ASSERT_EQ(host_output.copy_to_host(), cudaSuccess);
    
    // Verify: Last block should be false
    EXPECT_TRUE(host_output.at(0)) << "Block 0 should be true";
    EXPECT_FALSE(host_output.at(grid_size - 1)) << "Last block should be false (contains last element)";
}

TEST(AllTerminatedTest, IntegrationAllTrue) {
    // Test the complete all_terminated function with all true values
    const std::size_t n_elements = 256;
    const std::size_t block_size = 64;
    const std::size_t grid_size = (n_elements + block_size - 1) / block_size;
    
    HostBoolArray host_flags(n_elements, true);
    HostBoolArray host_buffer(grid_size, false);
    
    ASSERT_EQ(host_flags.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_buffer.copy_to_device(), cudaSuccess);
    
    DeviceBoolArray flags = host_flags.get();
    DeviceBoolArray buffer = host_buffer.get();
    
    bool result = all_terminated(flags, buffer, grid_size, block_size);
    
    EXPECT_TRUE(result) << "all_terminated should return true when all flags are true";
}

TEST(AllTerminatedTest, IntegrationOneFalse) {
    // Test the complete all_terminated function with one false value
    const std::size_t n_elements = 256;
    const std::size_t block_size = 64;
    const std::size_t grid_size = (n_elements + block_size - 1) / block_size;
    
    std::vector<bool> flags_data(n_elements, true);
    flags_data[123] = false;
    
    HostBoolArray host_flags(std::move(flags_data));
    HostBoolArray host_buffer(grid_size, false);
    
    ASSERT_EQ(host_flags.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_buffer.copy_to_device(), cudaSuccess);
    
    DeviceBoolArray flags = host_flags.get();
    DeviceBoolArray buffer = host_buffer.get();
    
    bool result = all_terminated(flags, buffer, grid_size, block_size);
    
    EXPECT_FALSE(result) << "all_terminated should return false when any flag is false";
}