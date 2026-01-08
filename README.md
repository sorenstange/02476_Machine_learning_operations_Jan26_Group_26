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

In this project we seek to train and evaluate different image classifiers in order to classify different types of rice. 
Our project will use the *Rice Image Dataset* from *Kaggle* (https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset). This dataset contains 75.000 different 250x250 pixels greyscale images of 5 different grains of rice (15.000 images of each type); namely: Arborio, Basmati, Ipsala, Jasmine & Karacadag. All the images are a photo of a single grain of rice with a corresponding label.
We are going to use PyTorch and PyTorch Lightning as the framework for defining and training our models. This will give us a framework where we have good control of the specifics of our code while reducing the amount of boilerplate code needed to be written, giving us more time to focus on our model architecture and hyper-parameter tuning.
We will use the open-source library PyTorch Image Models (TIMM) in our project, by selecting one or more pretrained model(s) from this library, to find which model architecture/type is best suited for classification of rice. We will definitly choose a ResNet model which will be compared to our own model architecture, which probably will be based on a convolutional neural network.

Our goals for this project will be:
    1) Avoid data-leakage.
    2) Minimize overfitting.
    3) Use profiling to optmize our code.
    4) Use logging software such as W&B to moniter the progress of our project.
    5) Get an accuracy above 90%.