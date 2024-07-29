from functools import reduce
import re
import glob
import math
from multiprocessing import Pool


# Korak 1: Preprocesiranje i Računanje TF za jedan tekst
def preprocess_text(text):
    text = reduce(lambda x, y: x.replace(y, ' '), ["'", "\n", ",", ".", "!", "?", '"', "(", ")", "#", "$", "@"], text)
    text = text.lower()
    return list(filter(lambda x: character_count(x) >= 3, re.findall(r'\b\w+\b', text)))


def character_count(word):
    return reduce(lambda acc, _: acc + 1, word, 0)


def compute_tf(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    words = preprocess_text(text)
    word_count = reduce(lambda acc, word: acc + 1, words, 0)
    word_freq = reduce(lambda acc, word: acc.update({word: acc.get(word, 0) + 1}) or acc, words, {})

    tf_values = map(lambda word_count_pair: (file_path, word_count_pair[0], word_count_pair[1] / word_count),
                    word_freq.items())

    return list(tf_values)


# Korak 2: Računanje TF za sve tekstove
def calc_tf_all_files(path_pattern):
    files = glob.glob(path_pattern)

    with Pool() as pool:
        tf_values_lists = pool.map(compute_tf, files)

    def combine_tf_lists(tf_list1, tf_list2):
        return tf_list1 + tf_list2

    all_tf_values = reduce(combine_tf_lists, tf_values_lists)

    return all_tf_values


# Korak 3: Računanje IDF Vrednosti
def calc_idf(all_tf_values, total_documents):
    # Kreiranje skupa svih jedinstvenih reči
    all_words = set(map(lambda x: x[1], all_tf_values))

    # Računanje frekvencije dokumenata za svaku reč
    doc_freq = reduce(
        lambda acc, word: acc.update(
            {word: reduce(lambda count, tf: count + (tf[1] == word), all_tf_values, 0)}) or acc,
        all_words,
        {}
    )

    # Računanje IDF za svaku reč
    idf_values = map(lambda word: (word, math.log(total_documents / doc_freq[word])), doc_freq)
    return dict(idf_values)


# Korak 4: Računanje TF-IDF Vrednosti
def calc_tf_idf(all_tf_values, idf_values):
    def calculate_tf_idf(file_path, word, tf):
        return word, file_path, tf * idf_values[word]

    tf_idf = list(map(lambda args: calculate_tf_idf(*args), all_tf_values))
    return sorted(tf_idf, key=lambda x: (x[1], -x[2]))


def main():
    file_paths = glob.glob('data/*.txt')
    all_tf_values = calc_tf_all_files('data/*.txt')
    total_documents = reduce(lambda acc, _: acc + 1, file_paths, 0)
    idf_values = calc_idf(all_tf_values, total_documents)
    tf_idf_values = calc_tf_idf(all_tf_values, idf_values)

    for file_path in file_paths:
        print(f"Dokument {file_path}:")

        # TF vrednosti
        tf_values = [tf for tf in all_tf_values if tf[0] == file_path]
        tf_dict = {word: round(tf, 4) for _, word, tf in tf_values}
        print("TF vrednosti:", tf_dict)

        # IDF vrednosti
        idf_dict = {word: round(idf_values[word], 4) for word in tf_dict.keys()}
        print("IDF vrednosti:", idf_dict)

        # TF-IDF vrednosti
        tf_idf_dict = [(word, round(tf_idf, 4)) for word, _, tf_idf in tf_idf_values if file_path in _]
        tf_idf_dict.sort(key=lambda x: -x[1])  # Sortiranje po vrednosti TF-IDF
        print("TF-IDF vrednosti:", tf_idf_dict)

        print("\n")

    print("Sorted list of (word, file name, TF-IDF value):")
    for word, file_path, tf_idf_value in tf_idf_values:
        print(f"{word} - {file_path} - {round(tf_idf_value, 4)}")


if __name__ == "__main__":
    main()