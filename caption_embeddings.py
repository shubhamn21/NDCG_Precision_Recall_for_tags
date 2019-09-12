import json
import nltk
from tqdm import tqdm
with open('/data/deva/mscoco/annotations/captions_train2014.json') as f:
    data = json.load(f)
print (data.keys())
exit()
#f = open("captions2.txt","w")
train_dict={}
val_dict={} 
for i in tqdm(data['annotations']):
    i['caption']= str(' '.join([nltk.stem.WordNetLemmatizer().lemmatize(word) for (word, pos) in nltk.pos_tag(nltk.word_tokenize(i['caption'])) if (pos[:2] == 'NN')])).lower()
    if i['image_id'] in train_dict.keys():    
        train_dict[i['image_id']].append(str(' '.join(list(set([word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(i['caption'])) if (pos[:2] == 'NN')])))).lower())
    else:
        train_dict[i['image_id']] = [str(' '.join(list(set([word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(i['caption'])) if (pos[:2] == 'NN')])))).lower()]
for k, v in train_dict.items():
     v=list(set((' '.join(v)).split()))
     f.writelines('COCO_train2014_'+str('%012d'%k)+'.jpg'+'\t\t["'+'","'.join(v)+'"]\n')
with open('/data/deva/mscoco/annotations/captions_val2014.json') as w:
    data = json.load(w)
for i in tqdm(data['annotations']):
    i['caption']= str(' '.join([word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(i['caption'])) if (pos[:2] == 'NN')])).lower()
    if i['image_id'] in val_dict.keys():
        val_dict[i['image_id']].append(str(' '.join(list(set([word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(i['caption'])) if (pos[:2] == 'NN')])))).lower())
    else:
        val_dict[i['image_id']] = [str(' '.join(list(set([word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(i['caption'])) if (pos[:2] == 'NN')])))).lower()]
for k, v in val_dict.items():
     v=list(set((' '.join(v)).split()))
     f.writelines('COCO_val2014_'+str('%012d'%k)+'.jpg'+'\t\t["'+'","'.join(v)+'"]\n')
f.close()
    
