#include "core/types.cuh"
#include "core/vec.cuh"
#include "propagation/propagate.cuh"
#include "simulation/simulation.cuh"
#include "utils.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <optional>

struct Errors {
  double position_error;
  double velocity_error;
  double epoch_error;

  static Errors from_simulation(const Simulation &simulation) {
    std::vector<double> state_error_components(simulation.n_samples() *
                                               STATE_DIM);
    std::transform(
        simulation.propagation_state.states.begin(),
        simulation.propagation_state.states.end(),
        simulation.expected_propagation_state.states_data.begin(),
        state_error_components.begin(),
        [](const double &a, const double &b) { return std::pow(a - b, 2.0); });
    std::vector<double> position_error_components(simulation.n_samples(), 0.0);
    for (auto vidx = 0; vidx < simulation.n_samples(); ++vidx) {
      for (auto dim = POSITION_OFFSET; dim < POSITION_OFFSET + POSITION_DIM;
           ++dim) {
        position_error_components[vidx] += state_error_components[get_2d_index(
            simulation.n_samples(), vidx, dim)];
      }
      position_error_components[vidx] =
          std::sqrt(position_error_components[vidx]);
    }
    double position_error = *std::max_element(position_error_components.begin(),
                                              position_error_components.end());

    std::vector<double> velocity_error_components(simulation.n_samples());
    for (auto vidx = 0; vidx < simulation.n_samples(); ++vidx) {
      for (auto dim = VELOCITY_OFFSET; dim < VELOCITY_OFFSET + VELOCITY_DIM;
           ++dim) {
        velocity_error_components[vidx] += state_error_components[get_2d_index(
            simulation.n_samples(), vidx, dim)];
      }
      velocity_error_components[vidx] =
          std::sqrt(velocity_error_components[vidx]);
    }
    double velocity_error = *std::max_element(velocity_error_components.begin(),
                                              velocity_error_components.end());

#ifndef NDEBUG
    print_vector_mean(position_error_components, "Position error components");
    print_failed_count(position_error_components, "Position error components",
                       simulation.tolerances.position);
    print_vector_mean(velocity_error_components, "Velocity error components");
    print_failed_count(velocity_error_components, "Velocity error components",
                       simulation.tolerances.velocity);
#endif

    std::vector<double> epoch_error_components(simulation.n_samples());
    std::transform(
        simulation.propagation_state.epochs.begin(),
        simulation.propagation_state.epochs.end(),
        simulation.expected_propagation_state.epochs.begin(),
        epoch_error_components.begin(),
        [](const double &a, const double &b) { return std::abs(a - b); });
    double epoch_error = *std::max_element(epoch_error_components.begin(),
                                           epoch_error_components.end());

    return Errors{position_error, velocity_error, epoch_error};
  }
};

bool validate(const Simulation &simulation) {
  const auto errors = Errors::from_simulation(simulation);
  const auto pos_passed =
      errors.position_error <= simulation.tolerances.position;
  const auto vel_passed =
      errors.velocity_error <= simulation.tolerances.velocity;
  const auto epoch_passed = errors.epoch_error <= simulation.tolerances.time;
  if (!pos_passed) {
    std::cout << "Position error: " << errors.position_error << " exceeds tolerance: " << simulation.tolerances.position << std::endl;
  } else {
    std::cout << "Position error: " << errors.position_error << " within tolerance: " << simulation.tolerances.position << std::endl;
  }
  if (!vel_passed) {
    std::cout << "Velocity error: " << errors.velocity_error << " exceeds tolerance: " << simulation.tolerances.velocity << std::endl;
  } else {
    std::cout << "Velocity error: " << errors.velocity_error << " within tolerance: " << simulation.tolerances.velocity << std::endl;
  }
  if (!epoch_passed) {
    std::cout << "Epoch error: " << errors.epoch_error << " exceeds tolerance: " << simulation.tolerances.time << std::endl;
  } else {
    std::cout << "Epoch error: " << errors.epoch_error << " within tolerance: " << simulation.tolerances.time << std::endl;
  }
  return pos_passed && vel_passed && epoch_passed;
}

int main() {

  const std::string file = "acceptance/halloumiconfig.json";
  std::cout << "Loading configuration from " << file << std::endl;
  auto configuration = json_from_file(file);
  auto simulation = Simulation::from_json(configuration);
  std::cout << "Loaded simulation with " << simulation.n_samples() << " samples" << std::endl;
  std::cout << "Propagating" << std::endl;
  propagate(simulation);
  if (simulation.propagated) {
    std::cout << "Propagated" << std::endl;
  } else {
    std::cout << "Propagation failed!" << std::endl;
    return 1;
  }

  if (validate(simulation)) {
    std::cout << "Validation successful!" << std::endl;
  } else {
    std::cout << "Validation failed!" << std::endl;
    return 1;
  }

  return 0;
}
