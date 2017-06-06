#%%
import numpy as np
#%%
from fairytext import character
from fairytext.character import NON_VALID_REGEX, NULL_CHAR, WHITESPACE_CHARS, WHITESPACE_INTS
from fairytext.containers import ShortVecs
#%%
import json
with open('data/SQuAD/train-v1.1.json', 'r') as f:
    train_set = json.load(f)
#%%

#%%
Questions_list = []
Answers_text_list = []
Answers_start_list = []
Answers_end_list = []
IDs_list = []
Context_list = []
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
        
#%%

#%%
ContextIntArrays = [character.str2ints(x) for x in Context_list]
QuestionIntArrays = [character.str2ints(x) for x in Questions_list]
#%%
ContextWhitespaceIndices = [np.argwhere(np.in1d(x, WHITESPACE_INTS)).ravel() for x in ContextIntArrays]
QuestionWhitespaceIndices = [np.argwhere(np.in1d(x, WHITESPACE_INTS)).ravel() for x in QuestionIntArrays]
#%%
AnswerStartIndices = np.asarray(Answers_start_list, dtype=np.int32)
AnswerEndIndices = np.asarray(Answers_end_list, dtype=np.int32)
