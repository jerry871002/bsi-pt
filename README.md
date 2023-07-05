# Kueitso Paper Implementation

This reposotory is the implementation of various BPR-based algorithms in Kueitso's latest research paper.

We implement six BPR-based algorithms

- Previous works
    - [BPR+](BPR&#32;Variants&#32;Papers/BPR+.pdf)
    - [Deep BPR+](BPR&#32;Variants&#32;Papers/DPN-BPR+.pdf)
    - [Bayes-ToM](BPR&#32;Variants&#32;Papers/Bayes-ToM.pdf)
    - [Bayes-OKR](BPR&#32;Variants&#32;Papers/Bayes-OKR.pdf)
- Our works
    - BSI
    - BSI-PT

In three environments

- [Grid World](src/grid_world/)
- [Navigation Game](src/navigation_game/)
- [Soccer Game](src/soccer_game/)

## Getting Started

To reproduce the results in the paper, run

```
scripts/run_exps_and_plot.sh [NUM_RUNS] [NUM_EPISODES]
```

the figures will be generated and store in `src/fig/`, the source data will be stored in `src/data/` and `src/csv/`.

If you want to run the experiments one by one or inspect each algorithm in detail, see [this document](src/README.md)