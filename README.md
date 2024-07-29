# TF-IDF Calculation Using Map-Reduce

Implementing TF-IDF (Term Frequency-Inverse Document Frequency) calculation following the Map-Reduce paradigm.

## Problem Descriptions

#### Data Preprocessing and TF Calculation for a Single Text
- **Function**: Compute the term frequency (TF) for words in a text file.
- **Details**: 
  - Replace apostrophes with spaces.
  - Remove special characters.
  - Convert text to lowercase.
  - Split text into words without using the built-in `split` function.
  - Discard words shorter than three characters.
  - Compute TF as the ratio of word occurrences to the total number of words.

#### TF Calculation for All Texts
- **Function**: Apply the TF calculation to all text files in the dataset.
- **Details**: 
  - Use `map` and/or `reduce` to consolidate TF values from all files into a single list.

#### IDF Calculation
- **Function**: Compute the inverse document frequency (IDF) for words across all texts.
- **Details**: 
  - Use `map` and/or `reduce` to calculate IDF values based on the number of documents containing each word.

#### TF-IDF Calculation
- **Function**: Calculate the TF-IDF values for words using the TF and IDF lists.
- **Details**: 
  - Use `map` and/or `reduce` to compute TF-IDF values.
  - Sort the result to list words by their TF-IDF values for each file.

## Running the Project
1. Ensure you have the required dependencies installed.
2. Place the dataset in the `data` directory.
3. Run `run.py` to start the TF-IDF calculation and display the results.

## Dependencies
- Python 3.x
- Required libraries: `functools`, `re`, `glob`, `math`, `multiprocessing`

## Notes
- The implementation avoids explicit loops and list/dictionary comprehensions.
- The length function `len` is implemented using `reduce`.
- Sorting is done using the built-in `sorted` function.
- Lambda functions are used for concise function definitions.
