# NLP-Sequence-Tagging

## Description
The `NLP-Sequence-Tagging` repository contains a Python-based tagging system for natural language processing tasks. It includes a script that reads training and test data, preprocesses the text, and calculates the probabilities necessary for a sequence tagging algorithm, achieving an accuracy of over 92%.

## Features
- Reads and parses multiple training and test datasets for sequence tagging.
- Preprocesses text data, tokenizing into sentences and words/tags.
- Employs statistical models to calculate initial, transition, and observation probabilities for tagging accuracy.

## How to Use
To use the tagging system, follow these steps:
1. Ensure Python and NumPy are installed on your system.
2. Place your training and test data files in the script's directory or specify their paths when running the script.
3. Execute the script with the command:
   ```bash
   python tagger.py --train train_file1.txt train_file2.txt --test test_file.txt --output output_file.txt
   ```
4. The tagged output will be saved to the specified output file.

## Requirements
- Python 3.x
- NumPy

## Accuracy
The system has been tested and achieves an accuracy of over 92% on the provided datasets.

## Acknowledgments
This project is part of an academic assignment in Natural Language Processing, demonstrating the efficacy of statistical models in sequence tagging tasks.
