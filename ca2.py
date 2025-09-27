import numpy as np
import math
import networkx as nx
import random
import matplotlib.pyplot as plt

def pagerank(graph):
    
    nodes = set()
    for key,values in graph.items():
        nodes.add(key)
        nodes.update(values)
    nodes = sorted(list(nodes))
    nodes_idx = {s:i for i,s in enumerate(nodes)}
    n=len(nodes)
    
    #adj 
    adj = np.zeros((n,n),dtype = int)
    for key,value in graph.items():
        key_idx = nodes_idx[key]
        for v in value:
            v_idx = nodes_idx[v]
            adj[key_idx][v_idx]=1
    
    # H
    H= np.zeros((n,n),dtype = float)
    for i in range(n):
        row_sum = np.sum(adj[i])
        if row_sum ==0:
            H[i] = np.ones(n)/n
        else:
            H[i] = adj[i]/row_sum
            
    #G
    transportation = np.ones((n,n))/n
    a = 0.8
    max_iter=100
    tol = 1e-5
    G = a*H +(1-a)*transportation
    
    #eigen
    enval,envec = np.linalg.eig(G.T)
    idx = np.argmax(np.abs(enval-1.0))
    vec = envec[:,idx].real
    if vec.sum()<0 : vec=-vec
    pr = vec/np.sum(vec)
    
    #iter
    ppr = np.ones(n)/n
    for i in range(max_iter):
        new_ppr = ppr @ G
        if np.allclose(new_ppr,ppr,atol=tol):
            print("converged")
            break
        ppr = new_ppr
    
    scores = {nodes[i]: ppr[i] for i in range(n)}
    
    g = nx.DiGraph([(k,v) for k,value in graph.items() for v in value])
    plt.figure(figsize=(10,8))
    pos = nx.spring_layout(g,seed=42)
    nx.draw(g,pos,arrows=True, with_labels=True, node_size = [scores.get(i,1.0)*5000 for i in g.nodes()])
    plt.show()
    
def get_shingles(docs,k=2):
    shingles=set()
    doc_shingle = []
    for doc in docs:
        text = doc.lower().split()
        s={"".join(text[i:i+k]) for i in range(len(text)-k+1)}
        shingles |= s
        doc_shingle.append(s)
    shingles = sorted(list(shingles))
    shingle_idx = {s:i for i,s in enumerate(shingles)}
    
    mat = np.zeros((len(shingles),len(docs)),dtype=int)
    for doc_id, text in enumerate(doc_shingle):
        for t in  text:
            sid = shingle_idx[t]
            mat[sid][doc_id]=1
    return mat,shingles

def minhash(mat,num_hash=100):
    num_s,num_d = mat.shape
    sig = np.full((num_hash,num_d), np.inf)
    
    p=2**31-1
    hash_func=[(random.randint(1,p-1),random.randint(0,p-1)) for _ in range(num_hash)]
    
    for r in range(num_s):
        row=mat[r]
        for h,(a,b) in enumerate(hash_func):
            val = (a*r+b)%p
            for c in range(num_d):
                if  row[c]==1:
                    sig[h][c] = min(val,sig[h][c])
    return sig

def minhashperm(mat,num_perm=100):
    num_s,num_d = mat.shape
    sig = np.full((num_perm,num_d), np.inf)

    for p in range(num_perm):
        perm = np.random.permutation(num_s)
        for c in range(num_d):
            for idx in perm:
                if mat[idx][c]==1:
                    sig[p][c] = idx
    return sig

def jacards(sig,i,j):
    return np.mean(sig[:,i]==sig[:,j])

def cosine(A):
    d=A.shape[1]
    sim = np.zeros((d,d),dtype=float)
    
    for i in range(d):
        for j in range(d):
            num = np.dot(A[:,i],A[:,j])
            den = (np.linalg.norm(A[:,i])*np.linalg.norm(A[:,j])+1e-5)
            sim[i][j] = num/den
    return sim

def euc(A):
    
    d=A.shape[1]
    sim = np.zeros((d,d),dtype=float)
    
    for i in range(d):
        for j in range(d):

            sim[i][j] = np.linalg.norm(A[:,i]-A[:,j])
    return sim

def tfidf(mat):
    v,d = mat.shape
    m = np.zeros((v,d),dtype=float)
    
    df = np.count_nonzero(mat>0, axis=1)
    for j in range(d):
        for i in range(v):
            tf = mat[i][j]/(np.sum(mat[:,j])+1e-6)
            idf = math.log((d+1)/(df[i]+1))+1
            m[i][j] = tf*idf
    return m

def get_matrix(docs):
    vocab=set()
    doc_text = []
    for doc in docs:
        text = doc.lower().split()
        for t in text:
            vocab.add(t)
        doc_text.append(text)
    vocab = sorted(list(vocab))
    vocab_idx = {s:i for i,s in enumerate(vocab)}
    
    mat = np.zeros((len(vocab),len(docs)),dtype=int)
    for doc_id, text in enumerate(doc_text):
        for t in  text:
            sid = vocab_idx[t]
            mat[sid][doc_id]+=1
    return mat,vocab
# ------------------ Example Usage ------------------

if __name__ == "__main__":
    docs = [
        "the quick brown fox",
        "the quick brown dog",
        "the fast brown fox"
    ]

    print("\n=== MinHash (Binary Shingle Matrix) ===")
    M, shingles = get_shingles(docs, k=3)
    print("Binary Shingle-Doc Matrix:\n", M)

    sig = minhash(M)

    # Hash function method
    for i in range(len(docs)):
        for j in range(i+1, len(docs)):
            sim = jacards(sig, i, j)
            print(f"Estimated Jaccard similarity (Doc {i}, Doc {j}): {sim:.3f}")


    # Permutation method
    sig_perm = minhashperm(M)
    print("\n--- Using Permutation Method ---")
    for i in range(len(docs)):
        for j in range(i+1, len(docs)):
            sim = jacards(sig_perm, i, j)
            print(f"Estimated Jaccard similarity (Doc {i}, Doc {j}): {sim:.3f}")



    print("\n=== TF-IDF (Cosine & Euclidean) ===")
    count_matrix, vocab = get_matrix(docs)
    print(count_matrix, vocab)
    mtfidf = tfidf(count_matrix)
    print(mtfidf)

    cos_sim = cosine(mtfidf)
    euc_dist = euc(mtfidf)

    print("Cosine Similarity:\n", cos_sim)
    print("Euclidean Distances:\n", euc_dist) 
graph = {'B': ['C'], 'C': ['A'], 'D': ['C']}
pagerank(graph)
