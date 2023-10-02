import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
from config import api_key
import pyjokes
import webbrowser
import string
import requests
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os
import openai
from config import api_key
from PyPDF2 import PdfReader

def fetch_text_from_pdf(pdf_link):
    reader=PdfReader(pdf_link)
    text=""
    for page in reader.pages:
        text+=page.extract_text()
    return text

def preprocess_text(text):
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
    return " ".join(tokens)

def collect_data():
    # Collect and preprocess data from the PDFs
    # make me a list of sentences

    preprocessedData = []
    # List of books on Chanakya Neeti with their PDF links
    books = [
        {"title": "Chanakya Neeti", "pdf_link": "Chanakya Neeti.pdf"},
        {"title": "Chanakya in Daily Life", "pdf_link": "Chanakya in Daily Life.pdf"}
    ]
    for book in books:
        pdf_link = book["pdf_link"]
        text = fetch_text_from_pdf(pdf_link)
        processed_text = preprocess_text(text)
        preprocessedData.extend(processed_text.split(".")[:-1])

    return preprocessedData

def create_model(sentences):
    vocab_size = 10000
    embedding_dim = 128
    max_seq_length = 50
    lstm_units = 256
    output_units = vocab_size

    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    input_sequences = []
    output_sequences = []

    for sentence_tokens in tokenized_sentences:
        for i in range(len(sentence_tokens) - 1):
            input_sequence = sentence_tokens[:i+1]  # Input sequence up to the (i+1)th word
            output_sequence = sentence_tokens[i+1]   # Output sequence from the (i+1)th word
            input_sequences.append(input_sequence)
            output_sequences.append(output_sequence)

    X_train= input_sequences
    y_train = output_sequences
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    # Pad sequences
    # X_train = pad_sequences(X_train, maxlen=max_seq_length)
    # X_val = pad_sequences(X_val, maxlen=max_seq_length)
    # y_val = pad_sequences(y_val, maxlen=max_seq_length)
    # y_train = pad_sequences(y_train, maxlen=max_seq_length)

    # print(X_train)
    # Build the model
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_seq_length),
        LSTM(lstm_units),
        Dense(output_units, activation='softmax')
    ])

    # Compile and train the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    # Save the trained model
    model.save('custom_llm_model.h5')

listener = sr.Recognizer()
engine = pyttsx3.init()

engine.setProperty('volume', 1.0)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


def talk(text):
    engine.say(text)
    engine.runAndWait()

chatStr =""
def chat(query):
    global chatStr
    print(chatStr)
    openai.api_key = api_key
    chatStr += f"Saman: {query}\n LiteAlexa: "
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt= chatStr,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    try:
        talk(response["choices"][0]["text"])
        chatStr += f"{response['choices'][0]['text']}\n"
        return response["choices"][0]["text"]
    except:
        pass

def ai(prompt):
    openai.api_key = api_key
    text = f"ChatGpt response for Prompt: {prompt} \n *************************\n\n"

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    try:
        text += response["choices"][0]["text"]
        if not os.path.exists("Openai"):
            os.mkdir("Openai")

        with open(f"Openai/{''.join(prompt.split('intelligence')[1:]).strip() }.txt", "w") as f:
            f.write(text)
    except:
        pass

def say(text):
    os.system(f'say "{text}"')

def take_command():
    try:
        with sr.Microphone() as source:
            print("listening....") 
            voice = listener.listen(source)
            command = listener.recognize_google(voice,language="en-in")
            command = command.lower()
            print(command)
    except:
        pass
    return command

def run_wife():
    command=take_command()
    sites = [["youtube", "https://www.youtube.com"], ["code forces", "https://codeforces.com"],["wikipedia", "https://www.wikipedia.com"], ["google", "https://www.google.com"],]
    for site in sites:
        if f"Open {site[0]}".lower() in command.lower():
            talk(f"Opening {site[0]} sir...")
            webbrowser.open(site[1])
            return 
    if 'play' in command:
        song=command.replace('play','')
        talk('playing'+song)
        pywhatkit.playonyt(song)

    elif 'time' in command:
        time=datetime.datetime.now().strftime('%I:%M %p')
        talk('Current time is '+ time)
        print(time)

    elif 'who is' in command:
        person = command.replace('who is','')
        info=wikipedia.summary(person,1)
        print(info)
        talk(info)

    elif 'date' in command:
        talk('Sorry, I am dating your dad Saman')

    elif 'joke' in command:
        jokee=pyjokes.get_joke()
        print(jokee)
        talk(jokee)

    elif 'start songs' in command:
        musicPath="D:\songs\Cheques.mp3"
        os.system(f"start {musicPath}")

    elif 'Using Artificial INtelligence'.lower() in command:
        ai(prompt=command)

    else:
        print("Chatting....")
        chat(command)

if __name__=='__main__':
    # while True:
    #     cnt=10
    #     while cnt:
    #         cnt=cnt-1
    #         run_wife()
    #     break
    create_model(collect_data())