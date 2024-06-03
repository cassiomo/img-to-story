from dotenv import find_dotenv, load_dotenv
from transformers import pipeline, set_seed
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
import os
import streamlit as st

load_dotenv(find_dotenv())
api_token = os.getenv("API_TOKEN")
print(api_token)

# img2text
def img2text(url):
    # Set max_length to control the length of the generated text
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    # Retrieve the generated text from the image
    text = image_to_text(url)[0]["generated_text"]

    print(text)
    return text

# llm
def generate_storyOpenAI(scenario):
    template = f"""
    You are a story teller:
    You can generate a short story based on a simple narrative, the story should be no more than 20 words:
    
    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["Scenario"])

    story_llm = LLMChain(llm=OpenAI(
        model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)
    # story = "a beautiful young woman in a black sema skirt and high heels dancing stock photo"
    # print(story)
    print("Using LLMChain OpenAPI")
    return story

def generate_story(scenerio):
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

    prompt = "scenerio"
    story = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

    print(story)
    return story

    # generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')
    # set_seed(42)
    #
    # prompt = "Once upon a time in a magical land"
    # # story = generator(prompt, max_length=100, num_return_sequences=1, truncation=True, pad_token_id=50256)[0][
    # #     'generated_text']
    #
    # story = generator(prompt, max_length=50, num_return_sequences=1, truncation=True, pad_token_id=50256)[0][
    #     'generated_text']
    #
    # print(story)


# text to speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {api_token}"}

    payloads = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)

    with open('audio.flac','wb') as file:
        file.write(response.content)

def main():
    st.set_page_config(page_title="im 2 audio story", page_icon="android.png")

    st.header("Turn img into audio story")
    uploaded_file = st.file_uploader("Choose an image ...", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()

        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Upload Image.',
                 use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_storyOpenAI(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio.flac")

# scenario = img2text("photo3.jpg")
# scenario = " abc"
# story = generate_story(scenario)
# text2speech(story)

if __name__ == '__main__':
    main()
