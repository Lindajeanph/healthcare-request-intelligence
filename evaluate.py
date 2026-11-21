import argparse, json
from collections import Counter
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from common import load_model,predict,parse_result

p=argparse.ArgumentParser()
p.add_argument('--adapter',default=None)
p.add_argument('--limit',type=int,default=100)
args=p.parse_args()
name = 'finetuned' if args.adapter else 'baseline'
with open('data/test.jsonl',encoding='utf-8') as f:
    rows = [json.loads(line) for line in f][:args.limit]
tok,model=load_model(args.adapter)
truth,preds=[],[]
valid=0; all_fields=0; priority_correct=0; action_correct=0; errors=[]
for row in rows:
    raw=predict(tok,model,row['request'])
    obj=parse_result(raw)
    expected=row['answer']
    truth.append(expected['category'])
    preds.append(obj.get('category','INVALID') if obj else 'INVALID')
    if obj:
        valid+=1
        priority_correct+=int(obj.get('priority')==expected['priority'])
        action_correct+=int(obj.get('action')==expected['action'])
        all_fields+=int(obj==expected)
    if obj!=expected and len(errors)<10:
        errors.append({'request':row['request'],'expected':expected,'output':obj,'raw':raw})
metrics={'n':len(rows),'category_accuracy':accuracy_score(truth,preds),
 'category_macro_f1':f1_score(truth,preds,labels=sorted(set(truth)),average='macro',zero_division=0),
 'valid_json_rate':valid/len(rows),'priority_accuracy':priority_correct/len(rows),
 'action_accuracy':action_correct/len(rows),'exact_match':all_fields/len(rows),
 'sample_errors':errors}
Path('outputs').mkdir(exist_ok=True)
with open(f'outputs/{name}_metrics.json','w') as f: json.dump(metrics,f,indent=2)
print(json.dumps({k:v for k,v in metrics.items() if k!='sample_errors'},indent=2))
print('Saved:',f'outputs/{name}_metrics.json')
