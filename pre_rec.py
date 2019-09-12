import json
import nltk 
import sys
from D16 import my_vocab

vocab = my_vocab(vocab_file = '/data/shubham/mscoco/captions_tags.txt')
v_d = vocab.word2idx
for k,v in v_d.items():
    v_d[k]=[0,0,0,0]
idx=0
f3=open('bleuscore.txt','w')
valdict={}
k=3
sent_prec,sent_recall,Total_sent,OTP,OFP,OTN,OFN,BIN,AVG_words_real,AVG_words_pred=0,0,0,0,0,0,0,0,0,0

with open('./real.txt','r') as f:
  with open('./pred.txt','r') as f2:
     for real_line,cap_line in zip(f,f2):
         valdict[idx]= [cap_line.strip().split(' ')]
         valdict[idx].append(real_line.strip().split(' '))

         #Check if word1 exists in vocab else "word1+word2" exists
         for j,word in enumerate(valdict[idx][0]):
             if word in v_d:
                 continue
             else:
                 valdict[idx][0][j]=word+' '+valdict[idx][0][j+1]
                 del valdict[idx][0][j+1]
         for j,word in enumerate(valdict[idx][1]):
             if word in v_d:
                 continue
             else:
                 valdict[idx][1][j]=word+' '+valdict[idx][1][j+1]
                 del valdict[idx][1][j+1]

         #Top K
         valdict[idx][0] = valdict[idx][0][:k]
         valdict[idx][1] = valdict[idx][1][:k]
 
         # Remove <end> and  '' if any
         if '<end>' in valdict[idx][0]: valdict[idx][0].remove('<end>')
         if '' in valdict[idx][0]: valdict[idx][0].remove('')
         if '<end>' in valdict[idx][1]: valdict[idx][1].remove('<end>')
         if '' in valdict[idx][1]: valdict[idx][1].remove('')
         if len(valdict[idx][1])==0 or len(valdict[idx][0])==0:
             continue
        
         wt_check=0
         TP,TN,FP,FN=0,0,0,0         
         for word in valdict[idx][0]:
                 #v_d_p[word]+=1
                 if(word in valdict[idx][1]):
                     v_d[word][0]+=1
                     TP+=1
                 else:
                     v_d[word][1]+=1
                     FP+=1
         for word in valdict[idx][1]: 
                 if(word in valdict[idx][0]):
                     continue
                 v_d[word][3]+=1
                 FN+=1
                 
         #FP = len(valdict[idx][0])-TP
         #FN = len(valdict[idx][1])-TP
         
         #0/1 Metric
         if(FN==0):
             BIN+=1

         # AVG number of words
         AVG_words_real+=len(valdict[idx][1])
         AVG_words_pred+=len(valdict[idx][0])

         # Precision Recall per sentence
         sent_prec+=TP/(TP+FP)
         sent_recall+=TP/(TP+FN) 

         Total_sent+=1
         #BLEUscore=nltk.translate.bleu_score.sentence_bleu([valdict[img_file][1]],valdict[img_file][0],weights=(1,0,0,0))
         OTP+=TP
         OFP+=FP
         OFN+=FN
         OTN+=TN
         print ("Sentence Precision: ",TP/(TP+FP),"\nRecall: ",TP/(TP+FN))
         print ("AVG sentence precision: ",float(sent_prec/Total_sent),"\nAVG Recall: ",float(sent_recall/Total_sent),"\nTotal sent: ",Total_sent)
         print ("TP: ",TP," FP: ",FP," FN: ",FN)
         print (valdict[idx][1],valdict[idx][0])
         idx+=1
print (idx)
results={}

for k1,v1 in v_d.items():
     if(v1[1]>0 or v1[3]>0) and (v1[0]>0): #V1= TP V3 = FP V2=FN
        #print ("Class: ",k1," ",v1/(v1+v3)," ",v1/(v1+v2))
        results[k1]=[v1[0]/(v1[0]+v1[1]),v1[0]/(v1[0]+v1[3]),v1[0],v1[1],v1[3]]
sorted_results = sorted(results.items(), key=lambda kv: kv[1][0],reverse=True)
print ("Class\t","\tPrecision\t","\tRecall\t","    TP\t","FP   ","FN")
for j in sorted_results:
    print (j)

#Calculate Class precision and recall
idx=0
precision=0
recall=0
for j in sorted_results[:]:
   if(j[1][0]!=0 and j[1][1]!=0):
      precision+=j[1][0] 
      recall+=j[1][1]
      idx+=1

print ("@ K =",k)
print ("0/1 Score: ",BIN/Total_sent,BIN,Total_sent)
print ("Avg real words per sentence= ",AVG_words_real/Total_sent,"\npred words= ",AVG_words_pred/Total_sent)
print ("Class Precision =",precision/idx,"\nRecall =",recall/idx)
print ("Number of labels: ",idx)
#print ("Sentence precision: ",float(sent_prec/Total_sent),"\nSentence Recall: ",float(sent_recall/Total_sent),"\nTotal sent: ",Total_sent)
print ("Overall Precision: ",OTP/(OTP+OFP),"\nOverall Recall: ",OTP/(OTP+OFN))
print ("TP: ",OTP," FP: ",OFP," FN: ",OFN,OTN)
