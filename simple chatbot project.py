from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random 
import numpy as np

def log_unknown_query(user_text, confidence):
    with open("unknown_logs.txt", "a", encoding="utf-8") as f:
        f.write(f"{user_text} | confidence={confidence}\n")



"""data = {
    "greeting" : ["hello", "hye", "good morning", "namaste", "Ram Ram"],
    "goodbye" : ["bye", "goodbye", "see you later", "take care"],
    "thanks" : ["thanks", "thankyou", "thx", "much appreciate", "gratefull"], 
    "about" : ["how are you", "what's your name? ", "what are you doing "]
}

sentences =["hello", "hye", "good morning", "namaste", "Ram Ram",
     "bye", "goodbye", "see you later", "take care",
     "thanks", "thankyou", "thx", "much appreciate", "gratefull", 
     "how are you", "what's your name? ", "what are you doing "]

labels = ["greeting", "greeting", "greeting", "greeting", "greeting",
     "goodbye", "goodbye", "goodbye", "goodbye",  "thanks",  "thanks",  "thanks",  "thanks", "thanks",
         "about" , "about" , "about" ]"""

data = {
    "greeting": ["hello", "hi", "hey", "hye", "hello there",
        "good morning", "good evening", "good afternoon",
        "namaste", "ram ram", "radhe radhe",
        "hi bot", "hey bot", "hello buddy",
        "kaise ho", "kya haal hai"
    ],

    "goodbye": [
        "bye", "goodbye", "bye bye", "see you",
        "see you later", "see you soon",
        "take care", "talk to you later",
        "catch you later", "i am leaving",
        "i have to go", "good night"
    ],

    "thanks": [
        "thanks", "thank you", "thankyou", "thx",
        "thanks a lot", "much appreciated",
        "really thanks", "thanks bro",
        "thanks buddy", "grateful",
        "appreciate it", "thanks for helping"
    ],

    "about": [
        "how are you", "how are you doing",
        "what is your name", "who are you",
        "tell me about yourself",
        "what are you doing", "what do you do",
        "are you a bot", "are you human",
        "what can you do", "how do you work"
    ],

    "fallback": [
        "Sorry, I didn't understand that 😕",
        "Can you rephrase your question?",
        "I'm still learning, please ask something else",
        "That seems new to me 🤔"
    ]

}

sentences = [
    # greeting
    "hello", "hi", "hey", "hye", "hello there",
    "good morning", "good evening", "good afternoon",
    "namaste", "ram ram", "radhe radhe",
    "hi bot", "hey bot", "hello buddy",
    "kaise ho", "kya haal hai",

    # goodbye
    "bye", "goodbye", "bye bye", "see you",
    "see you later", "see you soon",
    "take care", "talk to you later",
    "catch you later", "i am leaving",
    "i have to go", "good night",

    # thanks
    "thanks", "thank you", "thankyou", "thx",
    "thanks a lot", "much appreciated",
    "really thanks", "thanks bro",
    "thanks buddy", "grateful",
    "appreciate it", "thanks for helping",

    # about
    "how are you", "how are you doing",
    "what is your name", "who are you",
    "tell me about yourself",
    "what are you doing", "what do you do",
    "are you a bot", "are you human",
    "what can you do", "how do you work"
]

labels = [
    # greeting (16)
    "greeting", "greeting", "greeting", "greeting",
    "greeting", "greeting", "greeting", "greeting",
    "greeting", "greeting", "greeting", "greeting",
    "greeting", "greeting", "greeting", "greeting",

    # goodbye (12)
    "goodbye", "goodbye", "goodbye", "goodbye",
    "goodbye", "goodbye", "goodbye", "goodbye",
    "goodbye", "goodbye", "goodbye", "goodbye",

    # thanks (12)
    "thanks", "thanks", "thanks", "thanks",
    "thanks", "thanks", "thanks", "thanks",
    "thanks", "thanks", "thanks", "thanks",

    # about (11)
    "about", "about", "about", "about", "about",
    "about", "about", "about", "about", "about",
    "about"
]


Vectorizer  = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
x = Vectorizer.fit_transform(sentences)

model = model = LogisticRegression(max_iter=1000)
model.fit(x, labels)

model.fit(x, labels)

resposes = {
    "greeting" : ["hello! how can i help you😊","Hi there","Hi! Nice to see you ?"],
    "goodbye" : ["goodbye! have a great day😊","Bye","see you soon","Talk to you later"],
    "thanks" : ["You're welcome!","No problem","I'm Glad to help"],
    "about" : ["I am a simple ML-based chatbot"],
    "fallback": [
        "Sorry, I didn't understand that 😕",
        "Can you rephrase your question?",
        "I'm still learning, please ask something else",
        "That seems new to me 🤔"
    ]

}

#-----------------
#Single input & output (No loop)
#-------------------------------
user_input = input("Enter the message")
user_input_vector = Vectorizer.transform([user_input])
#predicted = model.predict(user_input_vector)[0]
probs = model.predict_proba(user_input_vector)
confidence = np.max(probs)
intent = model.classes_[np.argmax(probs)]

print("probablities", probs)
print("intent", intent)
print("confidence", confidence)

print(model.classes_)

#bot_reply = random.choice(resposes[intent])

"""if confidence<0.45:
       bot_reply = random.choice(resposes["fallback"])
       
       print(f"  You   : {user_input}")
       print(f"ChatGPT : {bot_reply}")"""


if confidence < 0.45:
    log_unknown_query(user_input, confidence)
    bot_reply = random.choice(resposes["fallback"])
    print(f"  You   : {user_input}")
    print(f"ChatGPT : {bot_reply}")



    #print("I don't understand about this question, clarify me!!!")
    
else:
   
    bot_reply = random.choice(resposes[intent])

    print(f"  You   : {user_input}")
    print(f"ChatGPT : {bot_reply}")
