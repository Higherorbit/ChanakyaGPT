import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes

listener = sr.Recognizer()
engine = pyttsx3.init()

engine.setProperty('volume', 1.0)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


def talk(text):
    engine.say(text)
    engine.runAndWait()

def take_command():
    try:
        with sr.Microphone() as source:
            print("listening....") 
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            print(command)
    except:
        pass
    return command

def run_wife():
    command=take_command()
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
    else:
        talk('Speak clearly fucker!')

while True:
    cnt=2
    while cnt:
        cnt=cnt-1
        run_wife()
    break