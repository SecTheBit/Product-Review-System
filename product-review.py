import os
import nltk
import nltk.corpus
nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import blankline_tokenize
from nltk.util import bigrams,trigrams,ngrams
#from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import sqlite3
import itertools
import pandas as pd
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import os
import pandas as pd
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
listed = drive.ListFile({'q': "title contains 'database.sqlite' "}).GetList()
for file in listed:
  print('title {}, id {}'.format(file['title'], file['id']))
download_path = os.path.expanduser('~/data')
try:
  os.makedirs(download_path)
except FileExistsError:
  pass
output_file = os.path.join(download_path, 'database.sqlite')
temp_file = drive.CreateFile({'id': '1YaTrGzVdNwMavTL36oTYN2hbnMkRyx-9'})
temp_file.GetContentFile(output_file)




conn=sqlite3.connect(output_file)
c=conn.cursor()
df=pd.read_sql_query("""select Text,Score,ProductId from Reviews WHERE Score !=3  and id%10==0 ORDER BY ProductId""",conn)
new_df=df[:100000]
actual_score=new_df['Score']
sorted_data = new_df.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')




final = sorted_data.drop_duplicates(subset = {"Text","Score"}, keep ='first', inplace=False)
final.loc[final['Score']>3,'Score']=10
final.loc[final['Score']<3,'Score']=20

final.loc[final['Score']==10,'Score']=1
final.loc[final['Score']==20,'Score']=0
final.head()


temp_df=final[final.Score==1].sample(25000)
#temp_df=final[final.Score==0].sample
temp_df2 = final[~final.index.isin(temp_df.index)]

temp_df2.head(10)


import re
alpha=out.to_string()
remove_tag=re.compile(r'<[^>]+>')
alpha=remove_tag.sub('',alpha)
AI1=""
for words in alpha:
    AI1=AI1+words.lower()
tokenizer=RegexpTokenizer(r'\w+')
AI2=tokenizer.tokenize(AI1)

from nltk.corpus import stopwords
set_stop=set(stopwords.words('english'))
new_list=[w for w in AI2 if not w in set_stop and len(w)>2 and w.isalpha()]

pst=PorterStemmer()
lemm_list=[]
lemmatizer=WordNetLemmatizer()
for words in new_list:
    #new_word=lemmatizer.lemmatize(words)
    new_word=pst.stem(words)
    if(len(new_word)>2):
        lemm_list.append(new_word)
    else:
        continue
lemm_list
lemm_list_copy=[]
for words in lemm_list:
  if words not in lemm_list_copy:
    lemm_list_copy.append(words)
    
positive_sentiments=[]
negative_sentiments=[]
cleaned_text=[]
temp_str=""
for sent in final['Text'].values:
  filtered_result=[]
  for tokens in lemm_list_copy:
    if(sent.find(tokens)>-1):
      if((final['Score'].values)[i]==1):
        positive_sentiments.append(tokens)
      else:
        negative_sentiments.append(tokens)
      
      if tokens not in filtered_result:
         filtered_result.append(tokens)
    else:
      continue
      
  tmp_str=" ".join(filtered_result)
  cleaned_text.append(tmp_str)
  i=i+1
final["cleaned_text"]=cleaned_text      



import numpy as np
from sklearn.neighbors import KNeighborsClassifier
def k_classifier_brute(X_train,Y_train):
  my_list=list(range(0,50))
  neighbours=(list(filter(lambda x: x%2!=0,my_list)))
  
  cv_scores=[]
              
  for k in neighbours:
              knn=KNeighborsClassifier(n_neighbors=k,algorithm="brute")
              scores=cross_val_score(knn,X_train,Y_train,cv=10,scoring='accuracy')
              cv_scores.append(scores.mean())
              
  MSE=[1-x for x in cv_scores]
  optimal_k=MSE.index(min(MSE))
  print("Optimal value of k= %s",neighbours[optimal_k]) # it is optimal k
  plt.plot(neighbours,MSE)
   
  for xy in zip(neighbours,np.round(MSE,3)) :
       plt.title("Misclassification Error vs K")
       plt.xlabel('Number of Neighbors K')
       plt.ylabel('Misclassification Error')
  plt.show()
  print("the misclassification error for each k value is :%s ", np.round(MSE,3))
    
  return optimal_k #it is optimal k
  
Y=final['Score']
Y.shape 
  
from sklearn .model_selection import train_test_split
X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
X_train_copy,Y_train_copy,x_test_copy=X_train,Y_train,x_test
print("shape of training set and test set is %s %s",X_train.shape,x_test.shape,Y_train.shape,y_test.shape)

#applying KNN on Bow
from sklearn.feature_extraction.text import CountVectorizer 
bow=CountVectorizer()
tfidf=TfidfVectorizer()
X_train=bow.fit_transform(X_train)
X_train_copy=tfidf.fit_transform(X_train_copy)


x_test=bow.transform(x_test)
x_test_copy=tfidf.transform(x_test_copy)

#!pip install sklearn --upgrade

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
optimal_k_1=k_classifier_brute(X_train,Y_train)
optimal_k_2=k_classifier_brute(X_train_copy,Y_train_copy)

#fitting the model with right value of k in bow
from sklearn.neighbors import KNeighborsClassifier
optimal_k_bow=KNeighborsClassifier(algorithm="brute",n_neighbors=23)
optimal_k_bow.fit(X_train,Y_train)


optimal_k_tfidf=KNeighborsClassifier(algorithm="brute",n_neighbors=49)
optimal_k_tfidf.fit(X_train_copy,Y_train_copy)

pred_bow=optimal_k_bow.predict(x_test)
pred_tfidf=optimal_k_tfidf.predict(x_test_copy)
#pred_acc=optimal_k_bow.score(X_train,Y_train)
#print("Accuracy of the Model =",pred_acc)

from sklearn import metrics
print(metrics.accuracy_score(y_test,pred_bow))
print(metrics.accuracy_score(y_test,pred_tfidf))

!pip install -U -q scikit-plot
import scikitplot.metrics as skplt
skplt.plot_confusion_matrix(y_test ,pred_bow)
skplt.plot_confusion_matrix(y_test,pred_tfidf)

x_test_new1=["build Quality is Very Premium looking Compact Design Fingerprint Sensor very Fast responsive Camera Performance Decent 64 GB variant already updated Android Pie Nice Experence","Camera not up to mark","awesome product","No worry Happy with your Purchase and good brand quality.","Best mobile in that price there was a stock Android and there Package headphone in that All over mobile was best.","I got it for 11999 on prime day it worths more than that lovely camera and performance no doubt with sd636 and too good ram management I dont know why some people are not liking it but its damn good","Nokia should just shut shop now going to return this piece of crap.","Very good performance and happy with this purchase","Its absolutely rubbish to update system file of 1306 MB & even after updated system file twice its showing repeated error couldn't update Installation problem","Worst phone","Super good phone fast phone.","Not able to start mobile since unpacking of box 2days before. Request to take back the mobile.","excellent","This phone is perfect and serves the purpose since there is more of calling and whatsapp usage only.I am happy with the performance of the phone.On time delivery and received sealed box overall good experience with flipkart seller and nokia","Another premium product from Nokia. Build quality and looks are superb. Stock android with clean interface not like customised android interfaces with adds. Bagged this amazing deal during prime days. Overall experience of the phone is awesome In this price band. Thanks amazon for delivering in a day almost. Hats off","Great phone and highly handy fits well in the hand. Camera is good and battery lasts for about a day and half. Nokia is really good still","Delivered today and surprised to see a black defective screen.","Only one word. Disappointing.","Incoming calls is not ringing properly and shows as a missed calls without ringing. torch has been on without any command. I think it is not working as a normal smartphone. I think any manufacturing defects in this smartphone. Please help me. How can I remove this problems."]
x_test_new_bow=bow.transform(x_test_new1)
x_test_new_tfidf=tfidf.transform(x_test_new1)
pred1=optimal_k_bow.predict(x_test_new_bow)
pred2=optimal_k_tfidf.predict(x_test_new_tfidf)

positive=0
negative=0
print(pred1)
print(pred2)
#prediction on basis of Bag of words
for i in pred1:
      if i==1:
         positive+=1
      else:
         negative+=1
print("Percentage of positive reviews is",negative/len(x_test_new1))
#
#
#print(pred[:40])


from sklearn.linear_model import LogisticRegression
logstic=LogisticRegression()
logstic.fit(X_train,Y_train)


#making predions
preds=logstic.predict(x_test)
print(X_train.shape,Y_train.shape)
print(x_test.shape,y_test.shape)

#making predions
preds=logstic.predict(x_test)
print(X_train.shape,Y_train.shape)
print(x_test.shape,y_test.shape)

positive_svm=0
negative_svm=0
pred_svm_bow=svc_classifier.predict(x_test_new_bow)
print(pred_svm_bow)
for i in pred_svm_bow:
  if i==1:
    positive_svm+=1
  else:
    negative_svm+=1
print("percentage of positive review is ",(positive_svm/len(x_test_new1))*100)
print("percentage of negative review is ",(negative_svm/len(x_test_new1))*100)







