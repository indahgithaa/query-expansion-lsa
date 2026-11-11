import numpy as np, qe_lsa.lsa as L
def expand_tok(model, query_tokens, n_terms=10):
    terms = [t for t in query_tokens if t in model.term_index]
    if not terms: return []
    c = model.vectors[[model.term_index[t] for t in terms]].mean(0); c /= np.linalg.norm(c)
    cos = model.vectors @ c; ex=set(query_tokens)
    cand = sorted([(model.vocab[j], float(cos[j])) for j in np.nonzero(cos>0)[0] if model.vocab[j] not in ex], key=lambda x:(-x[1],x[0]))
    return cand[:n_terms]
L.expand = expand_tok
import runpy; runpy.run_module("scripts.reproduce_4_12", run_name="__main__")
