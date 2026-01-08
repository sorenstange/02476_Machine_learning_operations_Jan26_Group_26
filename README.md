# rice_classifier

"This repository contains the solution for the project assignment of the DTU course 02476 Machine Learning Operations for group 26."

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).


## Project description

In this project we seek to train and evaluate different image classifiers in order to classify different types of rice. Our project will use the *Rice Image Dataset* from *Kaggle* (https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset). This dataset contains 75.000 different 250x250 pixels greyscale images of 5 different grains of rice; namely: Arborio, Basmati, Ipsala, Jasmine & Karacadag. We are going to use PyTorch as well as PyTorch Lightning for defining and training our models. We will use the open-source library PyTorch Image Models (TIMM) in our project, by selecting soma pretrained model from this library, to find which model architecture/type is best suited for classification of rice. We will choose a ResNet and this will be compared to our own model, which probably will be based on a convolutional neural network.