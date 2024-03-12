import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import re

def summary_generater(content):
  model_name="JordiAb/BART_news_summarizer"
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  inputs = tokenizer(content, return_tensors='pt')
  with torch.no_grad():
    summary_ids = model.generate(
      inputs['input_ids'],
      num_beams=4,
      max_length=250,
      early_stopping=True
      )
    summary = tokenizer.decode(
       summary_ids[0],skip_special_tokens=True)
  return summary


def calculatedayvalue(date_str):
    date_str = date_str.strip()
    target_date = datetime.strptime(date_str, "%Y/%m/%d")
    base_date = datetime(2002, 1, 1)
    base_value = 37257
    days_difference = (target_date - base_date).days
    target_value = base_value + days_difference
    return str(target_value)

def giveurl(date):
  url="https://timesofindia.indiatimes.com/2002/1/1/archivelist/year-2002,month-1,starttime-37257.cms"
  year,month,day=date.split("/")
  year=year.strip()
  new_url=url.replace("2002/1/1",date).replace("2002",year).replace("1",month).replace("37257",calculatedayvalue(date))
  return new_url

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity_score


def getcontent(url,summary):
    response=requests.get(url)
    if response.status_code == 200:
        soup=BeautifulSoup(response.content,'html.parser')
        links = soup.find_all('a')
        news_urls = [link.get("href") for link in links if "articleshow" in str(link.get("href"))]
        for news_url in news_urls:
                similar=0
                new_news_url =news_url.replace("/articleshow/", "/articleshowprint/")
                res=requests.get(new_news_url)
                soup1=BeautifulSoup(res.content,'html.parser')
                main_content = soup1.find_all('div', class_='Normal')
                content_text = ""
                for content_element in main_content:
                    content_text += content_element.get_text() + " "
                sumary=summary_generater(content_text)
                simliar=calculate_similarity(sumary,summary)
                if simliar > 0.80:
                  return "This news is a valid news 👍 and the url of link "+news_url
    return "This is a fake news 🤦‍♀️"

def getcontentupdated(url,summary):
    response=requests.get(url)
    if response.status_code == 200:
        soup=BeautifulSoup(response.content,'html.parser')
        links = soup.find_all('a')
        news_urls = [link.get("href") for link in links if "articleshow" in str(link.get("href"))]
        for news_url in news_urls:
                similar=0
                new_news_url = "https://timesofindia.indiatimes.com/"+news_url.replace("/articleshow/", "/articleshowprint/")
                res=requests.get(new_news_url)
                soup1=BeautifulSoup(res.content,'html.parser')
                main_content = soup1.find_all('div', class_='Normal')
                content_text = ""
                for content_element in main_content:
                    content_text += content_element.get_text() + " "
                sumary=summary_generater(content_text)
                simliar=calculate_similarity(sumary,summary)
                if simliar > 0.80:
                  return "This news is a valid news 👍 and the url of link "+" https://timesofindia.indiatimes.com "+news_url[1,"https://timesofindia.indiatimes.com"+news_url]
    return "This is a fake news 🤦‍♀️"

st.set_page_config(page_title="🤖🧠 News Validator")

with st.sidebar:
    st.title('🤖🧠 News Validator')
    st.write('🤖🧠 News validator checks credibility, accuracy, and reliability of news sources, ensuring factual reporting and combating misinformation.')
    st.write('Write the content along with Ana@t then date Example={content}An@nt{date}')
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today? Write a news content and date to validate?"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today? Write a news content and date to validate?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Validating..."):
            content,date=prompt.split("An@nt")
            content =re.sub(r'\s+', ' ', str(content)).strip()
            summary=summary_generater(content)

            st.write("Summary of given news content  :     " + summary)
            date=str(date)
            url=giveurl(date)
            year = int(date.split("/")[0])
            if year > 2023:
                response = getcontentupdated(url,summary)
                placeholder = st.empty()
                placeholder.markdown(response)
            else:
                response = getcontent(url,summary)
                placeholder = st.empty()
                placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)



