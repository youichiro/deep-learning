import os


id_to_word = {}
word_to_id = {}


def load_data(file_name, seed=1984):
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_name
    ja, en = [], []
