# project-template
Template repository your project
## Installation
- Clone the repo

```bash
git clone https://github.com/edouardkoehn/WM_Atlas.git
```
- Create your virtual env
```bash
conda create -n wm python=3
conda activate wm
```
- Install poetry
```bash
pip install poetry
```
- install the module and set up the precommit
```bash
poetry install
poetry run pre-commit install
poetry env info
```

## Managing dependencies
- to add an dependency with poetry for example numpy
```bash
poetry add numpy
```
It would automatically install and update the pyproject.toml file. If you didn't specify it, it would add it to the tool.poetry.dependencies.

- to add an dependency with poetry to a specific group
```bash
poetry add pytest --group test
```
It would automatically install and update the pyproject.toml file. The dependencies would be attached to the tool.poetry. dependencies.group.
[More info about poetry](https://python-poetry.org/docs/managing-dependencies/#adding-a-dependency-to-a-group)

## Pre-commit config
Pre-commit hooks are small software called when you do a git commit. Those hooks do several tests on your code (formatting, imports, etc. ). If not all the tests have been passed, the pre-commmit would not allow you to do the commit. You would have to resolve each of the error before beeing able to do the commit.

Good pratrice with pre-commit :
1) Git commit regularly
2) Never git commit when you are in a rush
