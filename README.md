# GWO – Practical Study (Two Problem Settings)

The main goal of this repository is to **study the usefulness of Grey Wolf Optimization (GWO)** (and, where implemented, **its variants**) by applying them to two different problem types:

1) **CNN hyperparameter tuning** for handwritten digit recognition (Python).
2) **JCAS multibeam optimization** for joint communication and sensing (MATLAB), including comparisons across multiple optimization algorithms.

## Repository Structure

- [CNN-Handwritten_Digit](CNN-Handwritten_Digit)
  - `cnn_basic.py`: baseline CNN (random or user-specified hyperparameters)
  - `cnn_gwo.py`: CNN + GWO tuning for `learning_rate` and `num_filters`
  - `compare_methods.py`: runs and compares baseline vs GWO-tuned CNN
  - `requirements.txt`: Python dependencies

- [Digital_Signal_Processing-JCAS_Multibeam](Digital_Signal_Processing-JCAS_Multibeam)
  - `main.m`: example script for generating/visualizing beams
  - `main_comparison.m`: benchmarking script comparing algorithms (Two-Step ILS, GWO, IGWO, ChaoticGWO, ...)
  - Supporting functions: `twoStepILS.m`, `GWO.m`, `IGWO.m`, `ChaoticGWO.m`, etc.

## What to look for (GWO focus)

- **CNN task (Python):** compare a baseline CNN run (random hyperparameters) vs **GWO-tuned** hyperparameters.
- **JCAS task (MATLAB):** compare convergence/fitness/time across **GWO and implemented variants** (e.g., IGWO, ChaoticGWO) and a reference method (Two-Step ILS).

## 1) CNN-Handwritten_Digit (Python)

### Requirements

- Python 3.x
- Install dependencies:

```
pip install -r CNN-Handwritten_Digit/requirements.txt
```

### Dataset layout

The scripts expect an `archive` folder structured by class label:

```
CNN-Handwritten_Digit/
  archive/
    0/  (jpg images)
    1/
    ...
    9/
```

Images are loaded as grayscale and resized to `28x28`.

### Run the comparison

Run the baseline CNN vs the GWO-tuned CNN:

```
python CNN-Handwritten_Digit/compare_methods.py
```

The scripts generate outputs such as `comparison_results.json`, plots (`results_*.png`), and saved models (`.h5`) depending on the run.

## 2) Digital_Signal_Processing-JCAS_Multibeam (MATLAB)

### Quick start

Open MATLAB and set the working directory to:

- [Digital_Signal_Processing-JCAS_Multibeam](Digital_Signal_Processing-JCAS_Multibeam)

Then run one of the following scripts:

- `main.m`: beam / multibeam generation demo
- `main_comparison.m`: runs and compares multiple optimization algorithms

### Source / attribution

The JCAS multibeam part and the Two-Step ILS-based workflow are adapted from (and inspired by) the public repository:

- https://github.com/RostyslavUA/jcas_multibeam_optimization

See the subfolder README for additional context and references:

- [Digital_Signal_Processing-JCAS_Multibeam/README.md](Digital_Signal_Processing-JCAS_Multibeam/README.md)
