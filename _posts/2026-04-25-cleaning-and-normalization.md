---
title: "NLP 텍스트 전처리 (2) - 정제와 정규화"
date: 2026-04-25 09:00:00 +0900
categories: [NLP, DL입문]
tags: [정제, 정규화]
math: true
media_subpath: /assets/img/posts/tokenization
---
토큰화 작업 전과 후에는 정제와 정규화라는 과정을 거쳐야 한다. 정제를 함으로써 노이즈 데이터를 제거할 수 있고, 정규화 과정을 거침으로써 표현 방법이 다른 언어들을 한꺼번에 처리할 수 있게 된다.  

## 1 정제
정제 과정에서 삭제해야 하는 노이즈 데이터는 특수문자 등일 수도, 자연어이기는 하지만 영 쓸 데가 없는 데이터들일 수도 있다. 영 쓸 데가 없는 데이터를 불용어라고 하는데, 일반적으로 영어의 관사나 한국어의 조사가 여기에 해당된다. 정제는 토큰화 작업 앞뒤로 이루어지며, 완벽한 정제는 쉽지 않기 때문에 적당한 합의점을 찾기도 한다.  
### 1.1 불용어
불용어는 상대적인 개념이지만, 일반적으로 등장 빈도가 너무 적은 단어나 길이가 짧은 단어(영어)의 경우 불용어로 취급한다. 한국어를 비롯한 한자문화권 내의 언어에서는 2~3글자 내외로 이루어진 단어들도 많은 의미를 갖지만 영어를 비롯한 라틴어 기반 문자의 경우에는 2글자보다 짧은 단어들은 대게 대명사나 관사들이기 때문에 제거하게 되는 것이다.  
  
nltk에서 제공하는 불용어 목록을 살펴보면 다음과 같다.  
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt

stop_words_list = stopwords.words('english')
print('불용어 개수: ', len(stop_words_list))
print('불용어 10개 출력: ', stop_words_list[:10])
```  
  
![nltk에서 제공하는 불용어](image2-1.png)
_nltk에서 제공하는 불용어 개수와 목록의 일부_  

교재에 제시된 것보다 늘어난 것으로 보아 주기적으로 업데이트 되는 듯하다.  

```python
example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example)

result = []
for word in word_tokens:
    if word not in stop_words:
        result.append(word)

print('불용어 제거 전: ', word_tokens)
print('불용어 제거 후: ', result)
```  
  
![불용어 제거 전과 후](image2-2.png)
_불용어 제거 전과 후_  
  
is, not, it, an 등이 불용어로 처리되는 것을 확인할 수 있다.  
  
불용어 사전을 따로 만들어 처리하는 방법도 있는데, 뉴스 기사의 일부를 예전에 과제에서 사용했던 불용어 사전을 이용해 정제해보았다.  
```python
okt = Okt()

example = "8일 전국 곳곳에 매서운 한파가 몰아치면서 낮 기온이 영하권으로 곤두박질치고 제주와 남부지방에는 폭설이 쏟아지는 등 기상 악화로 인한 피해가 속출했다. 항공기와 여객선 운항이 줄줄이 중단돼 발이 묶이는가 하면, 빙판길 교통사고와 수도계량기 동파 등 한파 피해도 잇달았다."
stop_words = {'하다', '이다', '되다', '아니다', '있다', '않다', '수', '들', '그리고', '하지만', '등', '위해', '때문', '에서', '으로'}

word_tokens = okt.morphs(example)

result = [word for word in word_tokens if not word in stop_words]

print('불용어 제거 전: ', word_tokens)
print('불용어 제거 후: ', result)
```  
  
```plaintext
불용어 제거 전:  ['8일', '전국', '곳곳', '에', '매서운', '한파', '가', '몰아치면서', '낮', '기온', '이', '영하', '권', '으로', '곤두박질', '치고', '제주', '와', '남부', '지방', '에는', '폭설', '이', '쏟아지는', '등', '기상', '악화', '로', '인한', '피해', '가', '속출', '했다', '.', '항공기', '와', '여객선', '운항', '이', '줄줄이', '중단', '돼', '발', '이', '묶이는가', '하면', ',', '빙판', '길', '교통사고', '와', '수도', '계량기', '동파', '등', '한파', '피해', '도', '잇', '달았다', '.']
불용어 제거 후:  ['8일', '전국', '곳곳', '에', '매서운', '한파', '가', '몰아치면서', '낮', '기온', '이', '영하', '권', '곤두박질', '치고', '제주', '와', '남부', '지방', '에는', '폭설', '이', '쏟아지는', '기상', '악화', '로', '인한', '피해', '가', '속출', '했다', '.', '항공기', '와', '여객선', '운항', '이', '줄줄이', '중단', '돼', '발', '이', '묶이는가', '하면', ',', '빙판', '길', '교통사고', '와', '수도', '계량기', '동파', '한파', '피해', '도', '잇', '달았다', '.']
```  
  
헌법을 처리할 때 사용했던 불용어 사전인데, 정제 결과를 뵌 텍스트의 유형 등에 따라 불용어 사전의 성격이 달라질 필요가 있을 듯하다.  

## 2 정규화
표현 방법이 다른 단어들을 묶어 같은 단어로 만들어주는 과정을 정규화라고 부른다. 다음과 같은 방법으로 정규화가 이루어진다.  
### 2.1 어간 추출과 표제어 추출
같은 의미를 가졌음에도 표기가 다른 언어를 통합하는 방법에는 어간 추출과 표제어 추출이 있다.  
  
어간은 용언(동사, 형용사)이 활용될 때 그 형태가 변하지 않고, 실질적인 의미를 갖는 부분이다. 한국어의 어간은 단독으로 사용될 수 없는 반면에, 영어에서는 어간이 하나의 단어 역할을 하기도 한다.  
  
한편 표제어는 사전에 등재되는 '원형'의 형태이다. 일반적으로 한국어는 어간에 어미 '-다'를 결합한 형태가 표제어이다. 영어는 단독으로 사용 가능한 어간이 곧 표제어로 쓰인다.  
#### 2.1.1 어간 추출
```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

sentence = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."

tokenized_sentence = word_tokenize(sentence)

print('어간 추출 전: ', tokenized_sentence)
print('어간 추출 후: ',[stemmer.stem(word) for word in tokenized_sentence])
```  
  
```plaintext
어간 추출 전:  ['This', 'was', 'not', 'the', 'map', 'we', 'found', 'in', 'Billy', 'Bones', "'s", 'chest', ',', 'but', 'an', 'accurate', 'copy', ',', 'complete', 'in', 'all', 'things', '--', 'names', 'and', 'heights', 'and', 'soundings', '--', 'with', 'the', 'single', 'exception', 'of', 'the', 'red', 'crosses', 'and', 'the', 'written', 'notes', '.']
어간 추출 후:  ['thi', 'wa', 'not', 'the', 'map', 'we', 'found', 'in', 'billi', 'bone', "'s", 'chest', ',', 'but', 'an', 'accur', 'copi', ',', 'complet', 'in', 'all', 'thing', '--', 'name', 'and', 'height', 'and', 'sound', '--', 'with', 'the', 'singl', 'except', 'of', 'the', 'red', 'cross', 'and', 'the', 'written', 'note', '.']
```  
  
어간 추출을 위한 stemmer에는 종류가 여러 가지가 있는데, 일정한 규칙을 바탕으로 추출하다 보니 그 값이 온전치 못하거나 엉뚱한 값이 나오기도 한다.  
  
```python
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

print('어간 추출 전: ', words)
print('포터 스테머: ', [porter_stemmer.stem(w) for w in words])
print('랭커스터 스테머: ', [lancaster_stemmer.stem(w) for w in words])
```  
  
```plaintext
어간 추출 전:  ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
포터 스테머:  ['polici', 'do', 'organ', 'have', 'go', 'love', 'live', 'fli', 'die', 'watch', 'ha', 'start']
랭커스터 스테머:  ['policy', 'doing', 'org', 'hav', 'going', 'lov', 'liv', 'fly', 'die', 'watch', 'has', 'start']
```  
  
같은 단어들을 분석하더라도 스테머의 종류에 따라 아주 다른 값이 나오기도 하는데, 그렇기 때문에 필요에 따라 알맞는 스테머를 선택할 필요가 있다.  
  
#### 2.1.2 표제어 추출
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

print('표제어 추출 전: ', words)
print('표제어 추출 후: ', [lemmatizer.lemmatize(word) for word in words])
```  
  
![표제어 추출 결과](image2-3.png)
_표제어 추출 결과_  
  
표제어 추출 결과도 영 매끄럽지 못하다. 이건 표제어 추출을 위한 lemmatizer가 품사 정보도 같이 받아야지 정확히 작동이 가능하기 때문이다.  
  
```python
print(lemmatizer.lemmatize('dies', 'v'))
print(lemmatizer.lemmatize('watched', 'v'))
print(lemmatizer.lemmatize('has', 'v'))
```  
  
![품사 정보를 포함](image2-4.png)
_품사 정보를 포함한 추출 결과_  
  
품사 정보를 같이 넣어주었더니 정상적으로 표제어가 나타나는 걸 볼 수 있다. 단어의 품사 정보를 함께 태깅해놓으면 표제어 추출을 원활하게 할 수 있을 것으로 보인다.  
  
### 2.2 대소문자 통합
영어권 언어에서 대소문자를 통합하는 것도 정규화의 일종인데, 특히 영어에서 대문자는 문장 맨 앞이나 고유명사 등 특정 상황에서만 쓰이기 때문에 소문자로 통합하는 작업을 거치게 된다. 소문자로 통합하게 되면 질의의 결과로 대문자를 사용한 것과 소문자를 사용한 것을 모두 사용할 수 있기 때문에 유용하다. 때에 따라 소문자로 변환하면 안되는 경우도 있기 때문에 모두 한꺼번에 통합해서는 안된다. 이러한 문제를 해결하기 위해 문장 맨 앞에 오는 글자만 소문자로 하고 나머지는 대문자로 하는 방법을 사용할 수도 있다. 머신러닝 시퀀스 모델을 이용하여 더 정확히 진행시킬 수 있지만, 이 역시 여러 문제를 수반한다. 그래서 일반적으로 예외 사항을 크게 고려하지 않고 모든 코퍼스를 소문자로 바꾸는 방법을 사용한다.  

> 독일어 같은 경우에는 대문자가 명사를 나타내는 중요한 표지 역할을 하는데, 이런 경우에는 대소문자 통합을 아예 하지 않을까? 언어별 개별 특성이 정규화 규칙을 압도해버린다면 어떻게 해야 할까?  
> 이거에 대해서 찾아봤는데, 독일어 문자 코퍼스는 대소문자 통합을 제한적으로 진행하거나 거의 진행하지 않는다고 한다. Stack Overflow의 답변에서도 독일어 코퍼스는 대문자를 유지하는 게 좋다고 나와있었다.
{: .prompt-info }