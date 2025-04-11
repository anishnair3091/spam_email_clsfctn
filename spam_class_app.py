#For web app creation

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from datasets import Dataset
import numpy as np
import evaluate
import streamlit as st
import pickle

st.write("""
# **Spam Emails/Messages Classification Application**

**Spam emails**, also referred to as junk email, spam mail, or simply spam, refers to unwanted/unrequested messages sent in bulk via email. 
Most email spam messages are commercial in nature. Whether commercial or not, many are not only annoying as a form of attention theft, but also dangerous because they may contain links that lead to phishing web sites or sites that are hosting malware or include malware as file attachments.

In many cases, the recipients are failing in classifying the SPAM and essential mails leading to them financial loss or legal consequences. 
And **this application is a solution** for the same. 

If you have received any such and need to know the authenticity, simply copy and paste that message in the text box provided in the left sidebar and the status will be displayed below[predition section] in a couple of minutes.

""")

st.sidebar.header('User input text box')

text= st.sidebar.text_input('Type or paste the message here',
              label_visibility= 'visible',
              disabled= False, placeholder=None)


st.write("""
    ### **The input text message for verifying the authenticity is below**""")
st.write(text)

data= pd.DataFrame.from_dict({'Message': [text]})


                 
# **Model Building**
#Read the csv file.
df = pd.read_csv('https://github.com/anishnair3091/spam_email_clsfctn/raw/refs/heads/main/Spam.csv',skiprows=1, header=0)

#Drop the rows with missing/NaN values
df.dropna(axis=1, inplace=True)

#Encoding the category types to numerical values
encoder= LabelEncoder()
encoder.fit(df['Category'])
df['label']= encoder.transform(df['Category'])

#Tokenization
tokenizer= BertTokenizerFast.from_pretrained('bert-base-uncased')

#Create Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels= 2)

#Prepare train, eval and test datasets
train_df = df.sample(200, random_state= 42)
eval_df = df.sample(200, random_state= 42)
test_df= df.iloc[4000:4100]

train_dataset= Dataset.from_pandas(train_df)
eval_dataset= Dataset.from_pandas(eval_df)
test_dataset= Dataset.from_pandas(test_df)

#Params for trainer
def tokenize_functions(example):
    return tokenizer(example['Message'], padding='max_length', truncation=True)
tokenized_datasets_train = train_dataset.map(tokenize_functions, batched=True)
tokenized_datasets_eval= eval_dataset.map(tokenize_functions, batched=True)
tokenized_datasets_test= test_dataset.map(tokenize_functions, batched=True)

#Training arguments
training_args= TrainingArguments(output_dir= 'test-trainer', eval_strategy= 'epoch')

metric = evaluate.load('accuracy')

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels= eval_pred
    predictions = np.argmax(logits, axis= -1)
    return metric.compute(predictions=predictions, references=labels)

trainer=Trainer(
    model= model,
    args= training_args,
    train_dataset= tokenized_datasets_train,
    eval_dataset= tokenized_datasets_eval,
    compute_metrics= compute_metrics
)

#Train the model
trainer.train()

#Predict the output

df_data= Dataset.from_pandas(data)

tokenized_df= df_data.map(tokenize_functions, batched=True)

pred= trainer.predict(tokenized_df)



x = pred.predictions[:, :1]

z = []
for number in x:
    if number < 0:
        y = 0
    else:
        y= 1

    z.append(y)

prediction= y

category_type= np.array(['Spam', 'Not Spam'])

predicted_output= category_type[prediction]

st.subheader('Prediction')
st.write(f'The message entered is {predicted_output}')

