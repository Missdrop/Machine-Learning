# Machine Learning

This is a repository for me to implement interesting algorithms I found during my study.
Includes **Traditional Machine Learning**, **Deep Learning** and **Reinforcement Learning**.

## Clone

This git repository includes submodules, so please add a `--recurse-submodules` option on clone or pull commands.

For example:

```shell
$ git clone --recurse-submodules git@github.com:Missdrop/Machine-Learning.git
```

Or use the command to sync submodules:

```shell
$ git submodule update --init --recursive
```

## Installation

This project uses [**uv**](https://github.com/astral-sh/uv) to manage packages and projects, the environment can be
easily
installed by:

```shell
$ uv sync --all-packages
```

> This command will automatically install torch+cu128 (CUDA 12.8) for Windows and linux users.
> If your GPU does not support [CUDA](https://developer.nvidia.com/cuda/toolkit) 12.8, please edit `pyproject.toml` to
> switch [torch version](https://pytorch.org/get-started/locally/).

## Run

This project widely uses Jupyter notebook, please make sure you have a Jupyter environment. (which is already installed if everything is OK)