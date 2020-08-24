from tkinter import *
from tkinter import filedialog
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import sys
import typing
import subprocess
from tkinter import scrolledtext
from dostoevsky.data import DataDownloader, DATA_BASE_PATH, AVAILABLE_FILES
import dostoevsky
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

def begin():
    file = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("all files", "*.*")))
    f = open(file)
    raw = f.read()
    sentences = nltk.sent_tokenize(raw)
    command = 'download'
    arguments = ['fasttext-social-network-model']
    if command == 'download':
        downloader = DataDownloader()
        for filename in arguments:
            if filename not in AVAILABLE_FILES:
                raise ValueError(f'Unknown package: {filename}')
            source, destination = AVAILABLE_FILES[filename]
            destination_path: str = os.path.join(DATA_BASE_PATH, destination)
            if os.path.exists(destination_path):
                continue
            downloader.download(source=source, destination=destination)
    else:
        raise ValueError('Unknown command')

    tokenizer = RegexTokenizer()
    tokens = tokenizer.split('всё очень плохо')  # [('всё', None), ('очень', None), ('плохо', None)]

    model = FastTextSocialNetworkModel(tokenizer=tokenizer)

    messages = sentences

    results = model.predict(messages, k=2)


    for message, sentiment in zip(messages, results):

        analysis_line = '\n', message, '\n', '->', '\n', sentiment, '\n'

        text.insert(END, analysis_line)

def count():
    file = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("all files", "*.*")))
    f = open(file)
    raw = f.read()
    sentences = nltk.sent_tokenize(raw)
    command = 'download'
    arguments = ['fasttext-social-network-model']
    if command == 'download':
        downloader = DataDownloader()
        for filename in arguments:
            if filename not in AVAILABLE_FILES:
                raise ValueError(f'Unknown package: {filename}')
            source, destination = AVAILABLE_FILES[filename]
            destination_path: str = os.path.join(DATA_BASE_PATH, destination)
            if os.path.exists(destination_path):
                continue
            downloader.download(source=source, destination=destination)
    else:
        raise ValueError('Unknown command')

    tokenizer = RegexTokenizer()
    tokens = tokenizer.split('всё очень плохо')  # [('всё', None), ('очень', None), ('плохо', None)]

    model = FastTextSocialNetworkModel(tokenizer=tokenizer)

    messages = sentences

    results = model.predict(messages, k=2)

    for message, sentiment in zip(messages, results):
        positive_values_all = [sentiment.get('positive') for message, sentiment in zip(messages, results)]
        positive_values = [0.0 if value == None else value for value in positive_values_all]

        negative_values_all = [sentiment.get('negative') for message, sentiment in zip(messages, results)]
        negative_values = [0.0 if value == None else value for value in negative_values_all]
    summary = (len(negative_values))
    text.insert(INSERT, summary)

def graph():
    file = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("all files", "*.*")))
    f = open(file)
    raw = f.read()
    sentences = nltk.sent_tokenize(raw)
    command = 'download'
    arguments = ['fasttext-social-network-model']
    if command == 'download':
        downloader = DataDownloader()
        for filename in arguments:
            if filename not in AVAILABLE_FILES:
                raise ValueError(f'Unknown package: {filename}')
            source, destination = AVAILABLE_FILES[filename]
            destination_path: str = os.path.join(DATA_BASE_PATH, destination)
            if os.path.exists(destination_path):
                continue
            downloader.download(source=source, destination=destination)
    else:
        raise ValueError('Unknown command')

    import dostoevsky
    from dostoevsky.tokenization import RegexTokenizer
    from dostoevsky.models import FastTextSocialNetworkModel

    tokenizer = RegexTokenizer()
    tokens = tokenizer.split('всё очень плохо')  # [('всё', None), ('очень', None), ('плохо', None)]

    model = FastTextSocialNetworkModel(tokenizer=tokenizer)

    messages = sentences

    results = model.predict(messages, k=2)

    for message, sentiment in zip(messages, results):
        positive_values_all = [sentiment.get('positive') for message, sentiment in zip(messages, results)]
        positive_values = [0.0 if value == None else value for value in positive_values_all]

        negative_values_all = [sentiment.get('negative') for message, sentiment in zip(messages, results)]
        negative_values = [0.0 if value == None else value for value in negative_values_all]
        summary = (len(negative_values))


    n_value = np.array(negative_values)
    p_value = np.array(positive_values)
    counts_value = np.arange(summary)
    plt.plot(counts_value, p_value, n_value)
    plt.show()

def choose():
    file = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("all files", "*.*")))
    f = open(file)
    raw = f.read()
    sentences = nltk.sent_tokenize(raw)
    sent1 = int(sentence_from.get())
    sent2 = int(sentence_to.get())
    result_sent = sentences[sent1 - 1: sent2]
    text.insert(INSERT, result_sent)

def begin_eng():
    file = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("all files", "*.*")))
    f = open(file)
    raw = f.read()
    sentences = nltk.sent_tokenize(raw)
    sid = SentimentIntensityAnalyzer()
    for sentence in sentences:
        ss = sid.polarity_scores(sentence)
        analysis_line_eng = '\n', sentence, '\n', '->', '\n', ss, '\n'
        text.insert(END, analysis_line_eng)

def count_eng():
    file = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("all files", "*.*")))
    f = open(file)
    raw = f.read()
    sentences = nltk.sent_tokenize(raw)
    sid = SentimentIntensityAnalyzer()
    positive_values = []
    for sentence in sentences:
        ss = sid.polarity_scores(sentence)
        pos_ss = ss.get('pos')
        positive_values.append(pos_ss)
    summary = len(positive_values)
    text.insert(INSERT, summary)

def choose_eng():
    file = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("all files", "*.*")))
    f = open(file)
    raw = f.read()
    sentences = nltk.sent_tokenize(raw)
    sent1 = int(sentence_from2.get())
    sent2 = int(sentence_to2.get())
    result_sent = sentences[sent1 - 1: sent2]
    text.insert(INSERT, result_sent)

def graph_eng():
    file = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("all files", "*.*")))
    f = open(file)
    raw = f.read()
    sentences = nltk.sent_tokenize(raw)
    sid = SentimentIntensityAnalyzer()

    positive_values = []

    for sentence in sentences:
        ss = sid.polarity_scores(sentence)
        pos_ss = ss.get('pos')
        positive_values.append(pos_ss)
    summary = len(positive_values)
    negative_values = []

    for sentence in sentences:
        ss = sid.polarity_scores(sentence)
        neg_ss = ss.get('neg')
        negative_values.append(neg_ss)

    n_value = np.array(negative_values)
    p_value = np.array(positive_values)
    counts_value = np.arange(summary)
    plt.plot(counts_value, p_value, n_value)
    plt.show()

window = Tk()
text = scrolledtext.ScrolledText(window, width=60, height=16, bg="#e6e6e6", wrap=WORD, fg='black', font=("Times", 12))
text.place(x=0,y=323)
window.title("Analyzer")
window.geometry('499x634')
window['bg'] = '#DCDCDC'
btn = Button(window, text='Выбрать текст и вывести анализ', bg="#b3b3b3", command=begin, height=3, width=30, font=("Fixedsys", 11))
btn.grid(column=1, row=0)
btn2 = Button(window, text='Посчитать количество предложений', bg="silver", height=3, width=30, font=("Fixedsys", 11), command=count)
btn2.grid(column=1, row=6)
btn4 = Button(window, text='Вывести предложения', bg="silver", height=3, width=30, font=("Fixedsys", 11), command=choose)
btn4.grid(column=1, row=2)
lbl1 = Label(window, text="            -           ", font=("Fixedsys", 11), bg = "#DCDCDC")
lbl1.grid(column=1, row=3)
sentence_from = Entry(window,width=8)
sentence_from.place(x=55,y=122)
sentence_to = Entry(window,width=8)
sentence_to.place(x=155,y=122)
btn5 = Button(window, text='Вывести график', bg="#D3D3D3", height=3, width=30, font=("Fixedsys", 11), command=graph)
btn5.grid(column=1, row=7)
btn6 = Button(window, text='Вывести значимые предложения', bg="#DCDCDC", height=3, width=30, font=("Fixedsys", 11))
btn6.grid(column=1, row=8)
btn8 = Button(window, text='Open file and show the analysis', bg="#b3b3b3", height=3, width=30, font=("Fixedsys", 11), command=begin_eng)
btn8.grid(column=2, row=0)
btn9 = Button(window, text='Count sentences', bg="silver", height=3, width=30, font=("Fixedsys", 11), command=count_eng)
btn9.grid(column=2, row=6)
btn10 = Button(window, text='Show sentences', bg="silver", height=3, width=30, font=("Fixedsys", 11), command=choose_eng)
btn10.grid(column=2, row=2)
lbl2 = Label(window, text="             -           ", font=("Fixedsys", 11), bg = "#DCDCDC")
lbl2.grid(column=2, row=3)
sentence_from2 = Entry(window,width=8)
sentence_from2.place(x=311,y=122)
sentence_to2 = Entry(window,width=8)
sentence_to2.place(x=400,y=122)
btn11 = Button(window, text='Show the plot', bg="#D3D3D3", height=3, width=30, font=("Fixedsys", 11), command=graph_eng)
btn11.grid(column=2, row=7)
btn12 = Button(window, text='Show the significant ones', bg="#DCDCDC", height=3, width=30, font=("Fixedsys", 11))
btn12.grid(column=2, row=8)
window.mainloop()