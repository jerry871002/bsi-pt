# BSI-PT

Welcome to the BSI-PT project! This project implements the BSI-PT (Bayesian Strategy Inference plus Policy Tracking) framework and the experiments presented in the paper [Opponent Exploitation Based on Bayesian Strategy Inference and Policy Tracking](https://ieeexplore.ieee.org/document/10148618).

In the experiments, we compare our BSI-PT framework with previous BPR variants including [BPR+](https://researchspace.csir.co.za/dspace/bitstream/handle/10204/9091/Hernandez-Leal_2016.pdf?sequence=1&isAllowed=y)[^bpr], [DPN-BPR+](https://drive.google.com/file/d/1FMxWLF3hAgKTomp-foczAY3ppfF-GE-2/view)[^dpn-bpr], and [Bayes-OKR](https://www.sciencedirect.com/science/article/abs/pii/S0950705122001605)[^okr] in the [extended batter vs. pitcher game (EBvPG)](environments/baseball_game.md) defined in the paper.

## Getting Started

To reproduce the results in the paper, use the `scripts/run_exps_and_plot.sh` script. There are two methods for setting up the environment to run the script.

### Virtual Environment

Make sure you have Python 3.8+ installed, then set up a virtual environment using `venv` and activate it.

```
python -m venv venv
source venv/bin/activate
```

Then install the dependencies in `requirements.txt` using `pip`.

```
pip install -r requirements.txt
```

### Docker

First, build the image. Make sure you are in the root directory of this repository.

```
docker build -t bsi-pt .
```

Run a container using the image and mount the `src/` directory to see the results.

```
docker run -it -v $(pwd)/src:/app/src bsi-pt
```

You will enter an interactive shell inside the container.

### Run the Experiments

Run the experiments using the following command:

```
scripts/run_exps_and_plot.sh [NUM_RUNS] [NUM_EPISODES]
```

`[NUM_EPISODES]` defines the length of each game, and `[NUM_RUNS]` defines the number of games to run.

The value of `[NUM_RUNS]` is quite large in the paper (200,000 runs) since we want to ensure the results are statistically significant. We set the default of `[NUM_RUNS]` to 1,000 in the project, which still requires a few minutes to complete. For a quick run, it's recommended to set it to a smaller number (e.g. 10). However, please note that the results may be quite unstable with a small number of runs.

The raw data will be stored in `src/data/` in `.pkl` format. The figures will be generated from those data and stored in `src/fig/`, then the data points in the figures will be stored in CSV format in `src/csv/`.

If you want to run the experiments individually or inspect each algorithm in detail, see [Run Individual Experiment](detail.md).

## Extras

In addition to the EBvPG environment, which is under the `baseball_game/` directory in this repository, we also implemented three other environments inspired by the Bayes-OKR[^okr] paper, including [Grid World](environments/grid_world.md), [Navigation Game](environments/navigation_game.md), and [Soccer Game](environments/soccer_game.md). In these three environments, [Bayes-ToMoP](https://arxiv.org/pdf/1809.04240)[^tom] and BSI (BSI-PT without policy tracking capability) agents are also presented.

*[BPR]: Bayesian Policy Reuse

[^bpr]: Hernandez-Leal, Pablo, et al. "Identifying and tracking switching, non-stationary opponents: A Bayesian approach." (2016).
[^dpn-bpr]: Zheng, Yan, et al. "Efficient policy detecting and reusing for non-stationarity in Markov games." (2021).
[^okr]: Chen, Hao, et al. "Accurate policy detection and efficient knowledge reuse against multi-strategic opponents." (2022).
[^tom]: Yang, Tianpei, et al. "Towards efficient detection and optimal response against sophisticated opponents." (2018).
