import json, unicodedata, re, collections
from qe_lsa import config
from qe_lsa.data import load_corpus
from qe_lsa.split import split_documents
from qe_lsa.text import remove_stopwords, tokenize
nfkc=0; u2060=0; uxx=0
for f in sorted(config.CHUNK_DIR.glob("*.json")):
    for r in json.loads(f.read_text(encoding="utf-8")):
        t=r["text"]; nfkc += unicodedata.normalize("NFKC",t)!=t; u2060 += t.count("\u2060"); uxx += bool(re.search(r"#U[0-9A-Fa-f]{4}",t))
print("NFKC changed",nfkc,"U+2060",u2060,"#Uxxxx chunks",uxx)
C=load_corpus()
ref=next(c for c in C.chunks["D029"] if c.cid=="D029:P0008:C000")
ns=remove_stopwords(ref.tokens); print("ref tokens",len(ref.tokens),len(set(ref.tokens)),"nosw",len(ns),len(set(ns)))
print("QA per type", collections.Counter(q.qtype for q in C.qas))
splits={s:split_documents(C,s) for s in config.SEEDS}
print("attempts",{s:p["attempt"] for s,p in splits.items()})
dev_tuning=set().union(*(splits[s]["dev"] for s in config.TUNING_SEEDS))
tot=ov=0
for s in config.SEEDS:
    units=[q for q in C.qas if q.doc_id in splits[s]["test"]]
    tot+=len(units); ov+=sum(q.doc_id in dev_tuning for q in units)
print("test units seeds1-10 (per scenario)",tot,"| from docs used for tuning (dev seeds1-3)",ov, f"{ov/tot:.1%}")
appear=collections.Counter(d for s in config.SEEDS for d in splits[s]["test"])
print("test-doc appearances: distinct docs",len(appear),"dist",sorted(collections.Counter(appear.values()).items()))
print("dev units seeds1-3:",sum(sum(q.doc_id in splits[s]['dev'] for q in C.qas) for s in config.TUNING_SEEDS)*2)
# baseline zero-score cases on dev seeds 1-3 (TPL)
z=0
for s in config.TUNING_SEEDS:
    for q in C.qas:
        if q.doc_id in splits[s]["dev"]:
            qt=set(remove_stopwords(tokenize(q.question)))
            if not any(qt & set(c.tokens) for c in C.chunks[q.doc_id]): z+=1
print("dev TPL units w/ no query term in report:",z)
