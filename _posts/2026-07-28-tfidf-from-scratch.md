---
title: "NLP 100문 100답 (1) TF-IDF를 직접 구현하기"
date: 2026-07-28 20:00:00 +0900
categories: [Study, NLP]
tags: [tf-idf, 정보검색, 100문100답]
math: true
---

원문: [100 NLP interview questions](https://medium.com/@milana.shxanukova15/100-nlp-interview-questions-c282190d73f4)
학습 목적으로 문항을 번역하고 답변은 직접 작성했습니다.

## Q1. Write TF-IDF from scratch.
### TF-IDF를 직접 구현해보세요.

```python
import pandas as pd
from math import log

docs = [..., ..., ..., ...]  # 문장 모음
vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()
N = len(docs)

def tf(t, d):
    return d.split().count(t)

def idf(t):
    df = sum(1 for doc in docs if t in doc.split())
    return log(N / (df + 1))

def tfidf(t, d):
    return tf(t, d) * idf(t)

result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns=vocab)
tf_
```

```python
result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index=vocab, columns=["IDF"])
idf_

result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tfidf(t, d))

tfidf_ = pd.DataFrame(result, columns=vocab)
tfidf_
```

- **DTM (Document Term Matrix)** : 문서 단어 행렬
- 여러 문서에 등장하는 단어의 빈도를 행렬로 정리한 것
- BoW의 확장(BoW: 문서 내 단어 빈도를 벡터로 나타낸 것)

- **TF-IDF**
    - 정의: 문서 안에서 어떤 단어가 얼마나 그 문서를 잘 대표하는지 나타내는 수치
    - 구성: TF와 IDF의 곱
        - TF(Term Frequency): 단어 출현 빈도
        - IDF(Inverse Document Frequency): 역문서빈도
