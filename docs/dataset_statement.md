# Dataset Statement

This repository references multiple public or academic datasets used for multimodal fake news detection research. The raw datasets are not bundled into this repository.

## Included in this repository

- code for training and evaluation
- documentation about dataset usage in the experiments
- references to the original dataset papers

## Not included in this repository

- full raw dataset files
- redistributed image archives
- processed dataset copies derived from restricted sources
- pretrained weights tied to restricted data unless separately cleared for release

## Datasets used in this project

### 1. Weibo

- paper source: Jin et al., ACM MM 2017
- task: Chinese social media rumor detection
- repository role: main experiment dataset

### 2. Gossip

- paper source: Shu et al., Big Data 2020
- task: English entertainment fake news detection
- repository role: main experiment dataset

### 3. CFND

- paper source: Zhang et al., IJCAI 2024
- task: Chinese cross-domain fake news detection
- repository role: main experiment dataset

## Access and licensing reminder

Before reproducing the experiments, please obtain each dataset through its official source, paper-linked release, or license-compliant academic access path.

Users of this repository are responsible for:

1. checking the license or usage restriction of each dataset
2. complying with redistribution rules
3. complying with privacy or platform-specific requirements
4. documenting any preprocessing changes made locally

## Recommended publication practice

When making this repository public, it is recommended to:

1. provide paper links for each dataset
2. avoid uploading full dataset archives directly unless redistribution is explicitly allowed
3. document the local directory structure expected by the training scripts
4. document any filtering, cleaning, or split reconstruction steps that affect final metrics

## Reproducibility note

If exact splits, file naming conventions, or image availability differ from the original experiments, results may shift noticeably. The following factors are especially important:

- train/validation/test split definitions
- label mapping
- missing image handling
- text cleaning rules
- language-specific tokenization

See [reproducibility.md](reproducibility.md) for the operational setup.
