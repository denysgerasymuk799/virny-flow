# Alpine Meadow

Alpine Meadow is an Interactive Automated Machine Learning tool.
What makes our system unique is not only the focus on interactivity, but also the combined systemic and algorithmic design approach.

We design novel algorithms for AutoML search and co-design the execution runtime to efficiently execute the ML workloads.
On one hand we leverage ideas from query optimization, on the other we devise novel selection and pruning strategies combining cost-based Multi-Armed Bandits and Bayesian Optimization.

## Installation

Please install SWIG 3.0. (SWIG 4.0 will fail because of`pyrfr`)

```bash
pip install -r requirements.txt
python setup.py install
```

To verify your installation, you can try the tests

```bash
pytest -s -v tests
```

## Examples

Please refer to the [optimizer tests](tests/test_optimizer.py) and [facade tests](tests/test_facade.py).

## Citation

```text
@inproceedings{shang2019democratizing,
  title={Democratizing data science through interactive curation of ml pipelines},
  author={Shang, Zeyuan and Zgraggen, Emanuel and Buratti, Benedetto and Kossmann, Ferdinand and Eichmann, Philipp and Chung, Yeounoh and Binnig, Carsten and Upfal, Eli and Kraska, Tim},
  booktitle={Proceedings of the 2019 International Conference on Management of Data},
  pages={1171--1188},
  year={2019}
}
```

## Contributors

* [Zeyuan Shang](http://www.shangzeyuan.com/)
* [Emanuel Zgraggen](http://emanuelzgraggen.com/)
* [Benedetto Buratti]()
* [Philipp Eichmann](http://cs.brown.edu/people/peichman/)
