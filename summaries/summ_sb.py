import requests,sys
from pathlib import Path

fn = Path(sys.argv[1])
txt = fn.read_text()

def summ(x, a, b, st):
    data = { "input": { "input": x[a:b] } }
    # informal; no sample
    sts = [{"url":"https://dashboard.scale.com/spellbook/api/v2/deploy/dx43cs7", "headers":{"Authorization":"Basic clfddrsso01goxq1aoort7pyp"}},
           {"url":"https://dashboard.scale.com/spellbook/api/v2/deploy/ec53cpo", "headers":{"Authorization":"Basic clfde6qsb000cu81adccgr1vw"}}]
    return requests.post(**sts[st], json=data).json()['output']

def txt_rng(i, x):
    j = i+16000
    t = x[i:j]
    return i + len(t) - t[-1:0:-1].find(' .') - 1

def summs(x, nm, st):
    i,j = 0,0
    res = ''
    while i!=txt_rng(i, x):
        j = txt_rng(i, x)
        res += summ(x, i, j, st) + '\n\n'
        print(i)
        i=j
    with open(fn.stem + f"-{nm}.txt", "w") as f: f.write(res)
    return res

res = summs(txt, 'summ0', 1)
print('--')
res = summs(res, 'summ1', 0)

