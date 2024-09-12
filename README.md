[![benchmark-coordination](https://github.com/letiziaia/benchmark-coordination/actions/workflows/validate.yml/badge.svg)](https://github.com/letiziaia/benchmark-coordination/actions/workflows/validate.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![codecov](https://codecov.io/gh/letiziaia/benchmark-coordination/graph/badge.svg?token=s4ZAsLnIim)](https://codecov.io/gh/letiziaia/benchmark-coordination)

# benchmark-coordination

This project is an integrated collection of network-based approaches for coordination detection, as found in the scientific literature. The coordination detection methods are abstracted and implemented in a modular and extensible way, allowing the common building blocks to be reused and combined in different ways.

## Structure

- `features_builder/`: functions to extract features from the raw data (e.g. text preprocessing, tf-idf, etc.)
- `network_builder/`: functions to create networks based on similarity, and to filter relevant edges
- `pipeline/`: abstractions to combine multiple steps into a single workflow
- `similarity_calculator/`: implementation of similarity scores
- `types/`: utilities for consistent code typing and datetime formatting
- `utils/`: general utilities that are re-used throughout the codebase (e.g. logging)
- `windowing/`: functions to slice the data (e.g. sliding time windows approaches)

## Coordination Detection Methods

| Reference                                                                                                                                                                                                                                                                   | Trace (Features)      | Filter                                        | Window Type                   | Metric (Threshold)                      | Community Detection / Classification | Detection Scores |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | --------------------------------------------- | ----------------------------- | --------------------------------------- | ------------------------------------ | ---------------- |
| [Pacheco, Diogo, et al. "Uncovering coordinated networks on social media: methods and case studies." Proceedings of the international AAAI conference on web and social media. Vol. 15. 2021.](https://doi.org/10.1609/icwsm.v15i1.18075)                                   | handle                | handle used by >= 2 users                     | entire time                   | cardinality                             | -                                    | -                |
| [Pacheco, Diogo, et al. "Uncovering coordinated networks on social media: methods and case studies." Proceedings of the international AAAI conference on web and social media. Vol. 15. 2021.](https://doi.org/10.1609/icwsm.v15i1.18075)                                   | image (RGB vector)    | users who shared >= 3 images                  | entire time                   | Jaccard similarity (>= 99th percentile) | -                                    | -                |
| [Pacheco, Diogo, et al. "Uncovering coordinated networks on social media: methods and case studies." Proceedings of the international AAAI conference on web and social media. Vol. 15. 2021.](https://doi.org/10.1609/icwsm.v15i1.18075)                                   | hashtag sequence      | users with >= 5 posts and >=5 unique hashtags | entire time                   | Jaccard similarity MinHash (> 0.8)      | -                                    | -                |
| [Pacheco, Diogo, et al. "Uncovering coordinated networks on social media: methods and case studies." Proceedings of the international AAAI conference on web and social media. Vol. 15. 2021.](https://doi.org/10.1609/icwsm.v15i1.18075)                                   | retweet               | -                                             | entire time                   | cosine similarity (> 0.9)               | -                                    | -                |
| [Pacheco, Diogo, et al. "Uncovering coordinated networks on social media: methods and case studies." Proceedings of the international AAAI conference on web and social media. Vol. 15. 2021.](https://doi.org/10.1609/icwsm.v15i1.18075)                                   | tweet time (30m bins) | -                                             | entire time                   | cosine similarity (> 0.9)               | -                                    | -                |
| [Vishnuprasad, Padinjaredath Suresh, et al. "Tracking fringe and coordinated activity on Twitter leading up to the US Capitol attack." Proceedings of the international AAAI conference on web and social media. Vol. 18. 2024.](https://doi.org/10.1609/icwsm.v18i1.31409) | retweet               | -                                             | time (len: , stride: )        | cardinality (> 1)                       | -                                    | -                |
| [Vishnuprasad, Padinjaredath Suresh, et al. "Tracking fringe and coordinated activity on Twitter leading up to the US Capitol attack." Proceedings of the international AAAI conference on web and social media. Vol. 18. 2024.](https://doi.org/10.1609/icwsm.v18i1.31409) | text                  | -                                             | activity (len: 10, stride: 1) | Ratclif-Obershelp similarity (> 0.7)    | -                                    | -                |
| [Hristakieva, Kristina, et al. "The spread of propaganda by coordinated communities on social media." Proceedings of the 14th ACM Web Science Conference 2022. 2022.](https://doi.org/10.1145/3501247.3531543)                                                              | retweet               | -                                             | entire time                   | cosine similarity                       | Louvain                              | -                |

## Development

![python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

This project is developed and tested on multiple python versions (3.11+). Dependencies are managed with [pipenv](https://github.com/pypa/pipenv#installation) and security vulnerabilities are scanned with `pip_audit`.

`pipenv` can be installed via `pip`:

```shell
$ pip install pipenv
```

Before running the project, the virtual environment, including development dependencies, needs to be installed and then activated:

```shell
# install dependencies (including dev)
$ pipenv install --dev

# activate environment
$ pipenv shell
```

The code in this repository follows PEP 8 style guide. Code can be formatted and linted with:

```bash
python -m black .
python -m ruff check .
```

Also see [Black](https://black.readthedocs.io/en/stable/index.html).

Additionally, static typing can be checked with:

```bash
python -m mypy
```

Unit tests are implemented with `pytest` and can be run with:

```bash
python -m pytest tests/
```

## License

[Apache License 2.0](LICENSE)
