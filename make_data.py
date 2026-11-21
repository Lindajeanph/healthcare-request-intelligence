import json, random
from pathlib import Path
from common import SYSTEM

random.seed(42)
Path('data').mkdir(exist_ok=True)
# Synthetic-only cases. Templates are intentionally simple for the first prototype.
scenarios = {
 'records_request': (['a clinic', 'a law office', 'an insurance reviewer', 'a patient representative'],
    ['copies of medical records', 'a discharge summary', 'a visit note', 'an imaging report'], 'route_to_roi_specialist'),
 'billing': (['a patient', 'a billing office', 'an employer', 'a customer'],
    ['an incorrect balance', 'a duplicate charge', 'an itemized invoice', 'a payment receipt'], 'route_to_billing'),
 'insurance': (['an insurance representative', 'a patient', 'a benefits coordinator', 'a clinic'],
    ['coverage verification', 'a prior authorization status', 'an eligibility question', 'an insurance claim status'], 'route_to_insurance'),
 'appointment': (['a patient', 'a front desk clerk', 'a family member', 'a scheduler'],
    ['a rescheduled visit', 'an appointment confirmation', 'a canceled appointment', 'an available appointment'], 'route_to_scheduling'),
 'technical_support': (['a nurse', 'a staff member', 'a remote worker', 'a coordinator'],
    ['a password reset', 'a locked account', 'a portal login error', 'an application access problem'], 'route_to_it'),
}
openers = ['Please assist with', 'I need help regarding', 'Can someone review', 'I am contacting support about',
           'I have a question about', 'Please route my inquiry about']
closers = ['Please advise on next steps.', 'Who handles this?', 'Could someone follow up?',
           'I am unsure who should review it.', 'Thank you for your help.']
items=[]
for category,(actors,topics,action) in scenarios.items():
    for i in range(100):
        actor = random.choice(actors)
        topic = random.choice(topics)
        urgent = i%5 == 0
        timing = ' This must be resolved before a scheduled handoff today.' if urgent else ''
        request = f'{random.choice(openers)} {topic} from {actor}.{timing} {random.choice(closers)}'
        answer = {'category':category,'priority':'urgent' if urgent else 'normal','action':action}
        items.append({'request':request,'answer':answer})
random.shuffle(items)
# Separate 400 training and 100 held-out examples; same template family means this is a smoke test, not a robust generalization test.
train,test=items[:400],items[400:]
for name,rows in [('train',train),('test',test)]:
    with open(f'data/{name}.jsonl','w',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row)+'\n')
print(f'Created {len(train)} train and {len(test)} test synthetic examples.')
