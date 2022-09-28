[![Documentation Status](https://readthedocs.org/projects/rouse/badge/?version=latest)](https://rouse.readthedocs.io/en/latest/?badge=latest)

rouse
=====

An implementation of the Rouse model of polymer dynamics. For [example usage](https://rouse.readthedocs.org/en/latest/examples/01_examples.html) and the full [API reference](https://rouse.readthedocs.org/en/latest/rouse.html) visit our documentation at [ReadTheDocs](https://rouse.readthedocs.org/en/latest)

To install `rouse` you can use the latest stable version from [PyPI](https://pypi.org/project/rouse)
```sh
$ pip install --upgrade rouse
```
or the very latest updates right from GitHub:
```sh
$ pip install git+https://github.com/OpenTrajectoryAnalysis/rouse
```

Developers
----------
Note the `Makefile`, which can be used to build the documentation (using
Sphinx); run unit tests and check code coverage; and build an updated package
for release with GNU `make`.

When editing the example notebooks,
[remember](https://nbsphinx.readthedocs.io/en/sizzle-theme/usage.html#Using-Notebooks-with-Git)
to remove output and empty cells before committing to the git repo.
[nbstripout](https://github.com/kynan/nbstripout) allows to do this
automatically upon commit.
