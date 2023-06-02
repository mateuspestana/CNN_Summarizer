import streamlit as st
import pandas as pd
import numpy as np
import os
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="CNN Summarizer", page_icon="🎩", layout="wide")

# Modelos
@st.cache_resource
def load_models():
    resumidor = pipeline('summarization', model='philschmid/flan-t5-base-samsum', max_length=160)
    classificador = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    tradutor = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-pt")
    return resumidor, classificador, tradutor

with st.spinner('Carregando modelos...'):
    resumidor, classificador, tradutor = load_models()

# Funções
def baixaCNN(url):
  r = requests.get(url).content
  soup = BeautifulSoup(r, 'lxml')
  materia = soup.find('div', {'class':'article__content'}).get_text(strip=True).replace(u'\xa0', u' ')
  return materia

def resume(texto):
  return resumidor(texto)[0]['summary_text']

def classifica(texto):
  return classificador(texto, ['Economia', 'Política', 'Cultura', 'Guerra'])['labels'][0]

def traduz(texto):
  return tradutor(texto)[0]['translation_text']

def faz_tudo(url):
  materia = baixaCNN(url)
  resumo = resume(materia)
  traducao = traduz(resumo)
  classificacao = classifica(resumo)
  return resumo, traducao, classificacao

st.title('CNN Summarizer')
st.markdown('## Sumarizador de notícias da CNN')

with st.form(key='url_form'):
    url = st.text_input(label='Insira a URL da notícia')
    submit_button = st.form_submit_button(label='Sumarizar')

if submit_button:
    with st.spinner('Sumarizando...'):
        try:
            resumo, traducao, classificacao = faz_tudo(url)
            col1, col2, col3 = st.columns(3)
            col1.markdown('### Resumo')
            col1.success(resumo)
            col2.markdown('### Tradução')
            col2.warning(traducao)
            col3.markdown('### Classificação')
            col3.info(classificacao)

        except Exception as e:
            st.error('Erro ao sumarizar')
            st.error(e)
