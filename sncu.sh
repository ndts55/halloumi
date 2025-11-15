#!/bin/bash
#SBATCH --job-name=hl_ncu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=45
#SBATCH --account=p0024741
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=out/ncu/%A.log
#SBATCH --error=out/ncu/%A.log

export STDOUT_LINE_BUFFERED=1

kernel_name=$1
ncu_output=$2

start=$(date +%s)

# Prepare environment
module purge
ml GCC CUDA

# Call NCU
sections="--section ComputeWorkloadAnalysis \
--section InstructionStats \
--section LaunchStats \
--section MemoryWorkloadAnalysis \
--section MemoryWorkloadAnalysis_Chart \
--section MemoryWorkloadAnalysis_Tables \
--section NumaAffinity \
--section Nvlink \
--section Nvlink_Tables \
--section Nvlink_Topology \
--section Occupancy \
--section PmSampling \
--section PmSampling_WarpStates \
--section SchedulerStats \
--section SourceCounters \
--section SpeedOfLight \
--section SpeedOfLight_HierarchicalDoubleRooflineChart \
--section SpeedOfLight_HierarchicalHalfRooflineChart \
--section SpeedOfLight_HierarchicalSingleRooflineChart \
--section SpeedOfLight_HierarchicalTensorRooflineChart \
--section SpeedOfLight_RooflineChart \
--section WarpStateStats \
--section WorkloadDistribution"
options="--launch-count 10 --source yes --import-source yes"

ncu_options="$options $sections"

echo "Running simulation with register_limit=$register_limit and block_size=$block_size"

if [ "$kernel_name" = "all" ]; then
	ncu $ncu_options -o "${ncu_output}" "./build/halloumi"
else
	ncu $ncu_options --kernel-name "$kernel_name" -o "${ncu_output}" "./build/halloumi"
fi

end=$(date +%s)

echo "Started at $(date -d @${start})"
echo "Ended at $(date -d @${end})"
echo "Duration: $((($end - $start) / 60)) minutes $((($end - $start) % 60)) seconds"
