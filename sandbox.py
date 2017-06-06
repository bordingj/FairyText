
import numpy as np
import json

from fairytext import character
from fairytext.character import NON_VALID_REGEX, NULL_CHAR, WHITESPACE_CHARS, WHITESPACE_INTS
from fairytext.containers import ShortVecs

def load_SQuAD_and_align(data_path='data/SQuAD/train-v1.1.json'):
    print('\nloading SQuAD data from {} ...'.format(data_path))
    with open(data_path, 'r') as f:
        train_set = json.load(f)
    Questions_list = []
    Answers_text_list = []
    Answers_start_list = []
    Answers_end_list = []
    IDs_list = []
    Context_list = []
    print('aligning Questions, Answers and Contexts...')
    for i in range(len(train_set['data'])):
        for item in train_set['data'][i]['paragraphs']:
            qas = item['qas']
            answers = [x['answers'][0] for x in qas]
            context = item['context']
            Context_list += len(qas)*[NON_VALID_REGEX.sub(NULL_CHAR, context),]
            Questions_list += [NON_VALID_REGEX.sub(NULL_CHAR,x['question']) for x in qas]
            Answers_text_list += [NON_VALID_REGEX.sub(NULL_CHAR,x['text']) for x in answers]
            Answers_start_list += [int(x['answer_start']) for x in answers]
            Answers_end_list += [int(x['answer_start'])+len(x['text']) for x in answers]
            IDs_list += [x['id'] for x in qas]
    return dict(
            context_texts=Context_list,
            question_texts=Questions_list,
            answer_texts=Answers_text_list,
            answer_start_indices=Answers_start_list,
            answer_end_indices=Answers_end_list,
            ids=IDs_list,
            )
    
def convert_aligned_squad_to_char_indices(aligned_data_set):
    print('\nconverting context texts characters to list of 1d integer arrays ...')
    ContextIntArrays = [np.asarray(character.str2ints(x),dtype=np.int16) for x in aligned_data_set['context_texts']]
    print('converting question texts characters to list of 1d integer arrays ...')
    QuestionIntArrays = [np.asarray(character.str2ints(x),dtype=np.int16)  for x in aligned_data_set['question_texts']]
    
    print('finding locations of whitespaces in context texts and question texts ..')
    ContextWhitespaceIndices = [np.argwhere(np.in1d(x, WHITESPACE_INTS)).ravel().astype(np.int16) for x in ContextIntArrays]
    QuestionWhitespaceIndices = [np.argwhere(np.in1d(x, WHITESPACE_INTS)).ravel().astype(np.int16) for x in QuestionIntArrays]
    
    return ((ContextIntArrays, ContextWhitespaceIndices), (QuestionIntArrays, QuestionWhitespaceIndices))

#%%
train_data = load_SQuAD_and_align(data_path='data/SQuAD/train-v1.1.json')

((ContextIntArrays, ContextWhitespaceIndices), 
 (QuestionIntArrays, QuestionWhitespaceIndices)) = convert_aligned_squad_to_char_indices(train_data)
#%%
AnswerStartIndices = np.asarray(train_data['answer_start_indices'], dtype=np.int32)
AnswerEndIndices = np.asarray(train_data['answer_end_indices'], dtype=np.int32)
#%%
