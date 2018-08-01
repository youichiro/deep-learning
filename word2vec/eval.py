import sys
sys.path.append('..')
from common.utils import most_similar
import pickle


pkl_file = 'cbow_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

while True:
    query = input('\nquery -> ')
    if query == 'end': break
    most_similar(query, word_to_id, id_to_word, word_vecs)
