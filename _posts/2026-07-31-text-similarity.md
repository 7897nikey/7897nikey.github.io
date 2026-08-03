---
title: "NLP 100문 100답 (7) 텍스트 유사도 측정"
date: 2026-07-31 09:10:00 +0900
categories: [NLP, 100문100답]
tags: [유사도, 코사인, 자카드]
math: true
---
원문: [100 NLP interview questions](https://medium.com/@milana.shxanukova15/100-nlp-interview-questions-c282190d73f4)
학습 목적으로 문항을 번역하고 답변은 직접 작성했습니다.  
## Q7. What metrics for text similarity do you know?
### 텍스트 유사도 측정 방법에는 어떤 것들이 있나요?

### 코사인 유사도
TF-IDF 등으로 벡터화한 후 유사도를 두 벡터 사이의 각도의 코사인으로 계산한다. (형태소 분석X)
### 자카드 유사도
텍스트 데이터를 집합으로 만든 뒤 교집합 크기를 합집합 크기로 나눈 값이다.  
문장 순서 뒤집어도 유사도가 100%로 나올 수 있으므로 의미의 유사도를 확인하기는 어렵다.
### 유클리디안 유사도 = L2 거리
### 맨하탄 유사도 = L1 거리