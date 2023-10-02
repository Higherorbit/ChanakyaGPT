import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes
import webbrowser

import os
import openai
from config import api_key

listener = sr.Recognizer()
engine = pyttsx3.init()

engine.setProperty('volume', 1.0)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


def talk(text):
    engine.say(text)
    engine.runAndWait()

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
        talk('Speak clearly fucker!')

if __name__=='__main__':
    while True:
        cnt=1
        while cnt:
            cnt=cnt-1
            run_wife()
        break