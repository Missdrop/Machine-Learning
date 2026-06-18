# Machine Learning

This is a repository for me to imply interesting algorithms I found during my study.
Includes **Traditional Machine Learning**, **Deep Learning** and **Reinforcement Learning**.

## Installation

This repository use [**uv**](https://github.com/astral-sh/uv) to manage packages, the environment can be easily
installed by:

```shell
$ uv sync
```

> This command will automatically install torch+cu132 (CUDA 13.2) for Windows and linux users.
> If your GPU do not support [CUDA](https://developer.nvidia.com/cuda/toolkit) 13.2, please edit `pyproject.toml` to
> switch [torch version](https://pytorch.org/get-started/locally/).

## Run

This project widely uses Jupyter notebook, please make sure you have a Jupyter environment.