#!/usr/bin/env bash
set -euo pipefail

# Run all combinations of correlation/kernel/sampling strategies using
# compare_estimation_method.py and save results to the specified folder.

# Environment tuning for predictable parallel runs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NCE_PARALLEL_BACKEND=processes

# Adjust these paths if needed
RESULTS_DIR="${HOME}/Projects/nonparametric_correlation_estimator_cubic/Results"
SCRIPT_DIR="${HOME}/Projects/nonparametric_correlation_estimator_cubic/Python"
SCRIPT_PATH="$SCRIPT_DIR/compare_estimation_method.py"


mkdir -p "$RESULTS_DIR"

# Activate conda environment 'cubic'
# Try to initialize conda for non-interactive shells, fallback to direct activate
if command -v conda >/dev/null 2>&1; then
    # Initialize conda for this shell session
    eval "$(conda shell.bash hook)" || true
    conda activate cubic
else
    echo "Warning: 'conda' not found in PATH. Please activate 'cubic' environment manually before running the script."
fi

# Define parameter lists
correlation_functions=(
  parabolic
  trigonometric_sine
  trigonometric_cosine
  cubic
  linear
  exponential
  semicircle_upper
  inverse_trig
  chebyshev_t3
)

kernel_functions=(
  gaussian
  epanechnikov
  triangular
  uniform
  quartic
  triweight
)

sampling_strategies=(uniform boundary)

# Fixed parameters
N_JOBS=16
H_GRID_START=-3
H_GRID_END=0
H_GRID_NUM=100
NS=(100 200 400 800 1600 3200)
N_REPEATS=200
BANDWIDTH_METHOD="mse"
NUMBER_TEST=100

# Run combinations
for corr in "${correlation_functions[@]}"; do
  for kern in "${kernel_functions[@]}"; do
    for samp in "${sampling_strategies[@]}"; do
      out_fname="${corr}_${kern}_${samp}_results.pkl"
      out_path="$RESULTS_DIR/$out_fname"

      echo "Starting: corr=$corr, kernel=$kern, sampling=$samp -> $out_path"

      python "$SCRIPT_PATH" \
        --n ${NS[@]} \
        --n_repeats $N_REPEATS \
        --correlation_function $corr \
        --kernel_function $kern \
        --sampling_strategy $samp \
        --n_jobs $N_JOBS \
        --h_grid_start $H_GRID_START \
        --h_grid_end $H_GRID_END \
        --h_grid_num $H_GRID_NUM \
        --bandwidth_selection_method $BANDWIDTH_METHOD \
        --number_test $NUMBER_TEST \
        --output_file "$out_path"

      echo "Finished: $out_path"
    done
  done
done

echo "All combinations finished. Results in: $RESULTS_DIR"
