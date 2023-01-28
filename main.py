import speech_recognition as sr     # For sppech recognition using google ai
import pyttsx3                      # For voice output
import os
import numpy as np                  # To generate random choice

from nlp_pipeline.chatbot import chatbot

class ChatBot():
    def __init__(self):
        print("----- Warming up -----")
    # Sets Name of chatbot
    def set_name(self,name):
        self.name = name
    # Retuens Name of chatbot
    def get_name(self):
        return self.name
    # Converts Speech to text
    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            print("Currently Listening...")
            recognizer.adjust_for_ambient_noise(mic,duration=1)
            audio = recognizer.listen(mic,timeout=15)
            text = "Error"
        try:
            text = recognizer.recognize_google(audio)
            print("Me -> ", text)
            return text
        except sr.RequestError as e:
            print("404 -> Could not request results; {0}".format(e))
            return text

        except sr.UnknownValueError:
            print("404 -> Unknown error occurred")
            return text
    # Converts text to speech
    def text_to_speech(self,text):
        print("AI -> ", text)
        speaker = pyttsx3.init()
        voice = speaker.getProperty('voices')
        speaker.setProperty('voice', voice[1].id)
        speaker.say(text)
        speaker.runAndWait()
    # Returnes NLP response
    def chat(self,text):
        chat = chatbot(text)
        return chat


if __name__ == "__main__":
    ai = ChatBot()

    while True:
        ai.text_to_speech("Do you want to chat or speak with me?")
        action = int(input("(1 to chat 2 to speak) \nMe -> "))
        # If user choose to chat
        if action == 1:
            inp = input("AI -> What do you want to call me?\nMe -> ")
            ai.set_name(name=inp)
            print("AI -> Great what's on your mind?")
            while True:
                inp = input("Me ->")
                if any(i in inp for i in ["quit","exit","close","shut down","bye"]):
                    break
                elif any(i in inp for i in ["your name","who are you"]):
                    print("AI -> I'm " + ai.get_name())
                else:
                    output = ai.chat(inp)
                    print("AI ->",output)
        # If user choose to speak
        elif action == 2:
            ai.text_to_speech("What do you want to call me?")
            while True:
                res = ai.speech_to_text()
                if res == "Error":
                    ai.text_to_speech("Sorry, come again?")
                else:
                    break
            ai.set_name(name=res)
            ai.text_to_speech("Great what's on your mind?")
            while True:
                res = ai.speech_to_text()
                if any(i in res for i in ["thank","thanks"]):
                    res = np.random.choice(["you're welcome!","anytime!","no problem!","cool!","I'm here if you need me!","mention not"])
                elif any(i in res for i in ["your name","who are you"]):
                    res = "I'm " + ai.get_name()
                    ai.text_to_speech(res)
                elif any(i in res for i in ["exit","close","quit","bye"]):
                    break
                else:   
                    if res=="Error":
                        res="Sorry, come again?"
                    else:
                        output = ai.chat(res)
                        ai.text_to_speech(output)
        # Good bye text
        res = np.random.choice(["Tata","Have a good day","Bye","Goodbye","Hope to meet soon","peace out!"])
        ai.text_to_speech(res)
        break
            