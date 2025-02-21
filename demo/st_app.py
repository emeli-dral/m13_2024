import streamlit as st
from topic_model import TopicClassifier

st.title("Text topic modeling")

st.markdown("This is a **demo** service")
st.markdown("This service uses RandomForest classification model and ConutVectorizer to detect the topic of the input text")

with st.form('form'):
    text_area = st.text_area('Enter the text here:', 'your text here')
    submit_button = st.form_submit_button('Detect')

    if submit_button: 
        classifier = TopicClassifier()
        topic = classifier.predict(text_area)
        st.write(f'Detected topic: {topic}')