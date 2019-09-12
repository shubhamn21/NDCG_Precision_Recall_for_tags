import numpy as np
import sys
def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum((np.power(2,r)-1) / np.log2(np.arange(2, r.size + 2)))
            #return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r_p,r, k, method=0):
    dcg_max = dcg_at_k(sorted(r_p, reverse=True), k, method)
    if not dcg_max:
        return 0.
    print (r_p,r)
    print (dcg_at_k(r, k, method),dcg_max)
    return dcg_at_k(r, k, method) / dcg_max


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    gt = open(sys.argv[1])
    pred = open(sys.argv[2])
    k = 10000
    y_real = list(gt.readline().rstrip('\n').split(' '))[:k]
    y_pred = list(pred.readline().rstrip('\n').split(' '))[:k]
    avg_ndcg=0
    ix=0
    for o in range(40000):
    #while y_real!=['']:
        
        if '<end>' in y_real: y_real.remove('<end>')
        if '' in y_real: y_real.remove('')
        if '<end>' in y_pred: y_pred.remove('<end>')
        if '' in y_pred: y_pred.remove('')
        if len(y_real)==0 or len(y_pred)==0:
             y_real = list(gt.readline().rstrip('\n').split(' '))[:k]
             y_pred = list(pred.readline().rstrip('\n').split(' '))[:k]
             continue
        ix+=1
        print("\n\nReal:      ",y_real)
        print("Predicted: ",y_pred)   
        tokenized_sents = [i for i in y_real]
        tokenized_sents = list(set(tokenized_sents))
        un_c=-1
        unk_sent=[]
        c = len(y_real)
        for idx,word in enumerate(y_pred):
           if word in tokenized_sents:
               y_pred[idx]=c-y_real.index(word)
           else:
               unk_sent.append(word)
               unk_sent = list(set(unk_sent))
               y_pred[idx] =  -unk_sent.index(word)
        print ("Unknown words",unk_sent)
        r_p=[]
        for idx,word in enumerate(y_real):
            r_p.append(c-idx)
        ndcg_score= ndcg_at_k(r_p,y_pred,1,1)
        avg_ndcg += ndcg_score 
        print("NDCG at k, (where k is len(real)):   ",ndcg_score)
        y_real = list(gt.readline().rstrip('\n').split(' '))[:k]
        y_pred = list(pred.readline().rstrip('\n').split(' '))[:k]
o+=1
print ("\n\n Avg NDCG: ",avg_ndcg/ix," ",ix)

