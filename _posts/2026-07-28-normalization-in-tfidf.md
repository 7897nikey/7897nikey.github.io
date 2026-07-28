---
title: "NLP 100문 100답 (2) TF-IDF에서의 정규화"
date: 2026-07-28 20:10:00 +0900
categories: [NLP, 100문100답]
tags: [tf-idf, 정보검색]
math: true
---
원문: [100 NLP interview questions](https://medium.com/@milana.shxanukova15/100-nlp-interview-questions-c282190d73f4)
학습 목적으로 문항을 번역하고 답변은 직접 작성했습니다.

## Q2. What is normalization in TF-IDF?
### TF-IDF에서 정규화란 무엇인가요?

긴 문서가 단어 빈도 점수에서 유리하지 않도록 벡터 값의 크기를 일정하게 맞추는 것으로, 크게 두 개의 층위로 나뉜다.  

#### 1. TF 정규화: 빈도의 조정
- **길이 정규화**: 원시 빈도를 문서의 전체 단어 수로 나눔
- **로그 스케일링**: 빈도가 10배라고 중요도가 10배는 아니므로 로그를 취함

$$
\text{tf}(t, d) = 1 + \log(\text{count}(t, d))
$$

sklearn에서는 `sublinear_tf=True` 옵션에 해당한다.


#### 2. 벡터 정규화: 완성된 문서 벡터의 크기를 조정

문서 벡터 \$$ \mathbf{x} = (x_1, x_2, \dots, x_n) $$ 에 대해:

- **L2 정규화 (Euclidean Norm)** — sklearn 기본값 `norm='l2'`  
$$
\hat{x}_i = \frac{x_i}{\sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}}
$$  
  정규화 후 벡터의 길이가 1 → 벡터의 내적 == 코사인 유사도
  문서 검색에서 L2를 기본으로 사용한다.

- L1 정규화 (Manhattan Norm)  
$$
\hat{x}_i = \frac{x_i}{|x_1| + |x_2| + \cdots + |x_n|}
$$  
  절댓값의 합 == 1
  TF-IDF 값은 0 이상이므로 각 원소가 전체에서 차지하는 비율로 계산된다.