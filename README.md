# NLP Assignment 2: Prefix Embeddings and Spelling Correction

This repository contains the implementation for **NLP Assignment 2**, focusing on handling text with spelling errors using **prefix embeddings** and **spelling correction** techniques. The goal is to enhance text classification performance in scenarios where input text contains typos or variations in spelling.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
  - [Installation](#installation)
  - [Running the Script](#running-the-script)
- [Example Output](#example-output)
- [Dependencies](#dependencies)
- [License](#license)

## Overview
The task addresses the problem of text classification in noisy input environments where typos are common. By incorporating prefix-based word embeddings and a spelling correction module, this implementation demonstrates:

1. Preprocessing text with error correction.
2. Representing words using prefix embeddings.
3. Training and evaluating a robust text classification model.

## Features
- **Prefix Embeddings:** Generates word representations using subword prefixes for better handling of typos.
- **Spelling Correction Module:** Implements a spelling correction mechanism to preprocess noisy text.
- **Robust Classification Pipeline:** Combines preprocessing and classification for end-to-end evaluation.
- **Evaluation Metrics:** Reports accuracy, precision, recall, and F1-score.

## Usage

### Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/sivaciov/NLP-A2.git
cd NLP-A2
```

### Running the Script
Ensure you have Python 3.6 or later installed. Run the main script as follows:

```bash
python prefix_classifier.py --train_file <path_to_training_data> --test_file <path_to_test_data> --correct_typos <true/false>
```

#### Command-line Arguments
- `--train_file`: Path to the training dataset (CSV or text format).
- `--test_file`: Path to the test dataset (CSV or text format).
- `--correct_typos`: Whether to apply spelling correction (default: `true`).

Example:
```bash
python prefix_classifier.py --train_file data/train.csv --test_file data/test.csv --correct_typos true
```

## Example Output
The script will output metrics for the test dataset:

Sample output:
```
Accuracy: 88.3%
Precision: 87.2%
Recall: 86.8%
F1-Score: 87.0%
```

## Dependencies
This implementation uses the following dependencies:
- `numpy`
- `torch`
- `nltk`
- `spacy`

Install the dependencies using:
```bash
pip install numpy torch nltk spacy
```

Ensure to download the `spacy` language model:
```bash
python -m spacy download en_core_web_sm
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to use, modify, and extend this project for your own purposes. Contributions are always welcome!
