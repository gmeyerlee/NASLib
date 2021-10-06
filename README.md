<p align="center">
  <a href="https://github.com/automl/NASLib">
    <img src="https://img.shields.io/badge/Python-3.7%20%7C%203.8-blue?style=for-the-badge&logo=python" />
  </a>&nbsp;
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/pytorch-1.9-orange?style=for-the-badge&logo=pytorch" alt="PyTorch Version" />
  </a>&nbsp;
  <a href="https://github.com/automl/NASLib">
    <img src="https://img.shields.io/badge/open-source-9cf?style=for-the-badge&logo=Open-Source-Initiative" alt="Open Source" />
  </a>
  <a href="https://github.com/automl/NASLib">
    <img src="https://img.shields.io/github/stars/automl/naslib?style=for-the-badge&logo=github" alt="GitHub Repo Stars" />
  </a>
</p>

# NAS-Bench-Suite
In this repository, we introduce NAS-Bench-Suite, a comprehensive and extensible collection of NAS benchmarks accessible through a unified interface, created with the aim to facilitate reproducible, generalizable, and rapid NAS research. NAS-Bench-Suite includes queryable NAS benchmarks such as NAS-Bench-101, NAS-Bench-201, NAS-Bench-301 (DARTS), NAS-Bench-NLP, NAS-Bench-ASR, TransNAS-Bench-101, NAS-Bench-MR, and more.

We use NAS-Bench-Suite to give an in-depth analysis of the generalizability of popular NAS algorithms and performance prediction methods, finding that many conclusions drawn from a few NAS benchmarks do not generalize to other benchmarks.


[**Setup**](#setup)
| [**Usage**](#usage)
| [**Docs**](examples/)
| [**Contributing**](#contributing)

# Setup

While installing the repository, creating a new conda environment is recomended. [Install PyTorch GPU/CPU](https://pytorch.org/get-started/locally/) for your setup.

```bash
conda create -n mvenv python=3.7
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

Run setup.py file with the following command, which will install all the packages listed in [`requirements.txt`](requirements.txt)
```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

To validate the setup, you can run tests:

```bash
cd tests
coverage run -m unittest discover -v
```

The test coverage can be seen with `coverage report`.

# Usage

To get started, check out [`demo.py`](examples/demo.py).

```python
search_space = SimpleCellSearchSpace()

optimizer = DARTSOptimizer(config)
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, config)
trainer.search()        # Search for an architecture
trainer.evaluate()      # Evaluate the best architecture
```

For more examples, see [naslib tutorial](examples/naslib_tutorial.ipynb) and [intro to search spaces](examples/search_spaces.ipynb).

## Contributing
:warning: this is an anonymized version of our codebase. 

In our future de-anonymized release, we will welcome contributions from the community, along with suggestions or comments. Users can create `pull requests` or open `issues`.
