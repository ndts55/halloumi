#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "core/types.cuh"
#include "propagation/prepare_simulation_run.cuh"
#include "propagation/cuda_utils.cuh"
#include "simulation/rkf_parameters.cuh"

class PrepareSimulationRunTest : public ::testing::Test {
protected:
    void SetUp() override {
        RKFParameters params;
        params.initial_time_step = 100.0;
        params.min_time_step = 1e-6;
        params.abs_tol = 1e-9;
        params.rel_tol = 1e-9;
        params.scale_state = 1.0;
        params.scale_dstate = 0.0;
        params.max_steps = 1000000;
        
        ASSERT_EQ(initialize_rkf_parameters_on_device(params), cudaSuccess);
        cudaDeviceSynchronize();
        
        initial_dt = params.initial_time_step;
    }
    
    double initial_dt;
};

TEST_F(PrepareSimulationRunTest, ForwardPropagation) {
    // end_epoch > start_epoch (forward propagation)
    const std::size_t n_samples = 10;
    const std::size_t block_size = 32;
    const std::size_t grid_size = (n_samples + block_size - 1) / block_size;
    
    HostFloatArray host_start_epochs(n_samples, 0.0);
    HostFloatArray host_end_epochs(n_samples, 1000.0);
    HostBoolArray host_termination_flags(n_samples, false);
    HostBoolArray host_simulation_ended(n_samples, false);
    HostBoolArray host_backwards(n_samples, true); // Initialize to wrong value
    HostFloatArray host_next_dts(n_samples, 0.0);
    
    ASSERT_EQ(host_start_epochs.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_end_epochs.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_termination_flags.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_simulation_ended.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_backwards.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_next_dts.copy_to_device(), cudaSuccess);
    
    prepare_simulation_run<<<grid_size, block_size>>>(
        host_end_epochs.get(),
        host_start_epochs.get(),
        host_termination_flags.get(),
        host_simulation_ended.get(),
        host_backwards.get(),
        host_next_dts.get()
    );
    
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    ASSERT_EQ(host_termination_flags.copy_to_host(), cudaSuccess);
    ASSERT_EQ(host_simulation_ended.copy_to_host(), cudaSuccess);
    ASSERT_EQ(host_backwards.copy_to_host(), cudaSuccess);
    ASSERT_EQ(host_next_dts.copy_to_host(), cudaSuccess);
    
    for (std::size_t i = 0; i < n_samples; ++i) {
        EXPECT_FALSE(host_termination_flags.at(i)) << "Sample " << i << " termination flag should be false";
        EXPECT_FALSE(host_simulation_ended.at(i)) << "Sample " << i << " simulation_ended should be false";
        EXPECT_FALSE(host_backwards.at(i)) << "Sample " << i << " backwards should be false (forward propagation)";
        EXPECT_DOUBLE_EQ(host_next_dts.at(i), initial_dt) << "Sample " << i << " next_dt should equal initial_time_step";
    }
}

TEST_F(PrepareSimulationRunTest, BackwardPropagation) {
    // end_epoch < start_epoch (backward propagation)
    const std::size_t n_samples = 10;
    const std::size_t block_size = 32;
    const std::size_t grid_size = (n_samples + block_size - 1) / block_size;
    
    HostFloatArray host_start_epochs(n_samples, 1000.0);
    HostFloatArray host_end_epochs(n_samples, 0.0);
    HostBoolArray host_termination_flags(n_samples, false);
    HostBoolArray host_simulation_ended(n_samples, false);
    HostBoolArray host_backwards(n_samples, false); // Initialize to wrong value
    HostFloatArray host_next_dts(n_samples, 0.0);
    
    ASSERT_EQ(host_start_epochs.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_end_epochs.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_termination_flags.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_simulation_ended.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_backwards.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_next_dts.copy_to_device(), cudaSuccess);
    
    prepare_simulation_run<<<grid_size, block_size>>>(
        host_end_epochs.get(),
        host_start_epochs.get(),
        host_termination_flags.get(),
        host_simulation_ended.get(),
        host_backwards.get(),
        host_next_dts.get()
    );
    
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    ASSERT_EQ(host_termination_flags.copy_to_host(), cudaSuccess);
    ASSERT_EQ(host_simulation_ended.copy_to_host(), cudaSuccess);
    ASSERT_EQ(host_backwards.copy_to_host(), cudaSuccess);
    ASSERT_EQ(host_next_dts.copy_to_host(), cudaSuccess);
    
    for (std::size_t i = 0; i < n_samples; ++i) {
        EXPECT_FALSE(host_termination_flags.at(i)) << "Sample " << i << " termination flag should be false";
        EXPECT_FALSE(host_simulation_ended.at(i)) << "Sample " << i << " simulation_ended should be false";
        EXPECT_TRUE(host_backwards.at(i)) << "Sample " << i << " backwards should be true (backward propagation)";
        EXPECT_DOUBLE_EQ(host_next_dts.at(i), -initial_dt) << "Sample " << i << " next_dt should be negative initial_time_step";
    }
}

TEST_F(PrepareSimulationRunTest, ResetsSimulationEndedFlag) {
    // simulation_ended is true initially, should be reset to false
    const std::size_t n_samples = 10;
    const std::size_t block_size = 32;
    const std::size_t grid_size = (n_samples + block_size - 1) / block_size;
    
    HostFloatArray host_start_epochs(n_samples, 0.0);
    HostFloatArray host_end_epochs(n_samples, 1000.0);
    HostBoolArray host_termination_flags(n_samples, false);
    HostBoolArray host_simulation_ended(n_samples, true); // Set to true
    HostBoolArray host_backwards(n_samples, false);
    HostFloatArray host_next_dts(n_samples, 0.0);
    
    ASSERT_EQ(host_start_epochs.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_end_epochs.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_termination_flags.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_simulation_ended.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_backwards.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_next_dts.copy_to_device(), cudaSuccess);
    
    prepare_simulation_run<<<grid_size, block_size>>>(
        host_end_epochs.get(),
        host_start_epochs.get(),
        host_termination_flags.get(),
        host_simulation_ended.get(),
        host_backwards.get(),
        host_next_dts.get()
    );
    
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    ASSERT_EQ(host_simulation_ended.copy_to_host(), cudaSuccess);
    ASSERT_EQ(host_termination_flags.copy_to_host(), cudaSuccess);
    
    for (std::size_t i = 0; i < n_samples; ++i) {
        EXPECT_FALSE(host_simulation_ended.at(i)) << "Sample " << i << " simulation_ended should be reset to false";
        EXPECT_FALSE(host_termination_flags.at(i)) << "Sample " << i << " termination flag should be false";
    }
}

TEST_F(PrepareSimulationRunTest, MixedForwardBackward) {
    // Mix of forward and backward propagation samples
    const std::size_t n_samples = 8;
    const std::size_t block_size = 32;
    const std::size_t grid_size = (n_samples + block_size - 1) / block_size;
    
    std::vector<double> start_epochs = {0.0, 1000.0, 500.0, 2000.0, 0.0, 1500.0, 300.0, 800.0};
    std::vector<double> end_epochs =   {1000.0, 0.0, 1500.0, 1000.0, 500.0, 500.0, 1000.0, 100.0};
    
    HostFloatArray host_start_epochs(std::move(start_epochs));
    HostFloatArray host_end_epochs(std::move(end_epochs));
    HostBoolArray host_termination_flags(n_samples, false);
    HostBoolArray host_simulation_ended(n_samples, false);
    HostBoolArray host_backwards(n_samples, false);
    HostFloatArray host_next_dts(n_samples, 0.0);
    
    std::vector<bool> expected_backwards = {false, true, false, true, false, true, false, true};
    
    ASSERT_EQ(host_start_epochs.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_end_epochs.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_termination_flags.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_simulation_ended.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_backwards.copy_to_device(), cudaSuccess);
    ASSERT_EQ(host_next_dts.copy_to_device(), cudaSuccess);
    
    prepare_simulation_run<<<grid_size, block_size>>>(
        host_end_epochs.get(),
        host_start_epochs.get(),
        host_termination_flags.get(),
        host_simulation_ended.get(),
        host_backwards.get(),
        host_next_dts.get()
    );
    
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    
    ASSERT_EQ(host_backwards.copy_to_host(), cudaSuccess);
    ASSERT_EQ(host_next_dts.copy_to_host(), cudaSuccess);
    ASSERT_EQ(host_termination_flags.copy_to_host(), cudaSuccess);
    ASSERT_EQ(host_simulation_ended.copy_to_host(), cudaSuccess);
    
    for (std::size_t i = 0; i < n_samples; ++i) {
        EXPECT_EQ(host_backwards.at(i), expected_backwards[i]) 
            << "Sample " << i << " backwards flag mismatch";
        
        double expected_dt = expected_backwards[i] ? -initial_dt : initial_dt;
        EXPECT_DOUBLE_EQ(host_next_dts.at(i), expected_dt) 
            << "Sample " << i << " next_dt mismatch";
        
        EXPECT_FALSE(host_termination_flags.at(i)) 
            << "Sample " << i << " termination flag should be false";
        EXPECT_FALSE(host_simulation_ended.at(i)) 
            << "Sample " << i << " simulation_ended should be false";
    }
}