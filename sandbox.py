#%%

#%%
from fairytext import character
from fairytext.character import NON_VALID_REGEX
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
IDs_list = []
Context_list = []
for i in range(len(train_set['data'])):
    train_set['data'][i]['paragraphs']
    for item in train_set['data'][i]['paragraphs']:
        qas = item['qas']
        answers = [x['answers'][0] for x in qas]
        context = item['context']
        Context_list += len(qas)*[NON_VALID_REGEX.sub('\0', context),]
        Questions_list += [NON_VALID_REGEX.sub('\0',x['question']) for x in qas]
        Answers_text_list += [NON_VALID_REGEX.sub('\0',x['text']) for x in answers]
        Answers_start_list += [int(x['answer_start']) for x in answers]
        IDs_list += [x['id'] for x in qas]
        
#%%
ContextVecs = ShortVecs([character.str2ints(x) for x in Context_list])
QuestionVecs = ShortVecs([character.str2ints(x) for x in Questions_list])