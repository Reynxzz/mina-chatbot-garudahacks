import streamlit as st
import nltk
#nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('model_mina.h5')
import json
import random

intents = json.loads(open('intentsMinaLarge.json').read())
words = pickle.load(open('texts.pkl','rb'))
labels = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.50
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
        return return_list 
        
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            if tag == 'types_perfectionist':
                followup_perfectionist()
                result = ""
                break
            elif tag == 'just_talk':
                generative_model()
            result = random.choice(i['responses'])
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

def chatbot_response_direct(msg):
    return msg

def followup_perfectionist():
    st.success("Mina: You could be one of the people who have imposter syndrome, the Perfectionist one. May I ask you a few questions?")
    message = st.text_input("Have you ever been accused of being a micromanager?")
    if message == 'yes':
        message = st.text_input("Do you have great difficulty delegating? Even when you're able to do so, do you feel frustrated and disappointed in the results?")
        if message == 'yes':
          message = st.text_input("When you miss the (insanely high) mark on something, do you accuse yourself of “not being cut out” for your job and ruminate on it for days?")
          if message == 'yes':
            message = st.text_input("Do you feel like your work must be 100% perfect, 100% of the time?")
            if message == 'yes':
                st.success("Mina: Thats indicate you are the Perfectionist one. I have a mission to conquer your symptoms, wanna join?")
                if st.button(f"Let's go to mission!"):
                    st.image('perfectionist_welcome.jpg')
                    st.markdown('<div style="text-align: justify;">Perfectionism and imposter syndrome often go hand-in-hand. Think about it: Perfectionists set excessively high goals for themselves, and when they fail to reach a goal, they experience major self-doubt and worry about measuring up. Whether they realize it or not, this group can also be control freaks, feeling like if they want something done right, they have to do it themselves.</div>', unsafe_allow_html=True) 
                    st.markdown('<div style="text-align: justify;"></div>', unsafe_allow_html=True) 
                    st.write('*Please follow the mission below to overcome this symtomps!*')
                    perfect_mission1()            
            else:
                if message != "":
                    st.success("Mina: Thats good! maybe you just need someone to talk")  
          else:
            if message != "":
                st.success("Mina: Thats good! maybe you just need someone to talk")
        else:
            if message != "":
                st.success("Mina: Thats good! maybe you just need someone to talk")
    else:
        if message != "":
            st.success("Mina: Thats good! maybe you just need someone to talk")

def perfect_mission1():
    st.markdown('### **Mission 1: Done is better than perfect**')
    st.image('perfectionist_mission1.jpg')
    lst = ['Note down the advantages and disadvantages of being a perfectionist. Whenever you find yourself falling back into perfectionism, take another look at the disadvantages and move on', 'Set achievable goals for yourself. Setting attainable goals will keep you from pursuing unattainable perfection. This way, you can achieve your goals with the resources you have', 'Set time limits for tasks and make sure to follow them. To avoid spending excess time trying to perform a task perfectly, create a realistic time limit and stick to it']
    for i in lst:
        st.markdown("- " + i)
    st.button('Done? submit your mission here!')

# GENERATIVE MODEL CHATBOT
# --------------------------------------------------------------- #
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
def load_data():    
 tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
 model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
 return tokenizer, model

def generative_model():
    tokenizer, model = load_data()

    st.write("###### You can talk to me anything you say. I hope i can understand you :)")
    input = st.text_input('Talk to me anything:')
    if 'count' not in st.session_state or st.session_state.count == 6:
        st.session_state.count = 0 
        st.session_state.chat_history_ids = None
        st.session_state.old_response = ''
    else:
        st.session_state.count += 1

    new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.count > 1 else new_user_input_ids

    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    if st.session_state.old_response == response:
        bot_input_ids = new_user_input_ids
    
    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    st.write(f"Mina: {response}")

    st.session_state.old_response = response

# --------------------------------------------------------------- #

# STREAMLIT APP
st.markdown('<style>body{text-align:center;background-color:black;color:white;align-items:justify;display:flex;flex-direction:column;}</style>', unsafe_allow_html=True)
st.title("Mina: Your Personal Mentor to Conquer Imposter Syndrome")
st.image('landingPageMina.jpg')
st.markdown("Mina is a chatbot that will help you conquer your imposter syndrome. You can talk about what symptoms of imposter syndrome you want to overcome then Mina will direct you to structured missions")
st.image('landingPageMina2.jpg')
#print("bot is live")
message = st.text_input("You can ask me anything about imposter syndrome, or just share your feelings with me!")
st.markdown('<div style="text-align: justify; font-size: 10pt"><b>Tips:</b> you can ask the FAQ of Imposter Syndrom, like what is imposter syndrome, how to overcome this syndrome, what causes this, and many more!</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify;></div>', unsafe_allow_html=True)
ints = predict_class(message, model)
res = getResponse(ints,intents)
if res != "":
    st.success("Mina: {}".format(res))