import streamlit as st
import pytesseract
from PIL import Image
from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import CountVectorizer
# import pytesseract
import shutil
import os
import random
# import cv2
try:
  from PIL import Image
except ImportError:
  import Image


# Sklearn models
from sklearn.pipeline import Pipeline  # a pipeline is a sequence of data processing components or steps, where the output of one step is fed as the input to the next step
from sklearn.linear_model import LogisticRegression  # regressio model
from sklearn.neighbors import KNeighborsClassifier  # classifier
from sklearn.tree import DecisionTreeClassifier  # decision tree classifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier ,GradientBoostingClassifier # classfier
import neattext.functions as nfx
from sklearn.model_selection import train_test_split
import pandas as pd
## model building
df = pd.read_csv("Constraints1.csv")  # reading data
newdf=df
df.drop("id", axis=1, inplace=True)
df["label"] = df.label.map({"real":0, "fake":1})
df['Clean_Text'] = df['tweet'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)
df = df.drop('tweet', axis=1)
Xfeatures = df['Clean_Text']
ylabels = df['label']
x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=42)


# loaded_model.fit(x_train,y_train)

# TEXT BASED PREDICTION  testing
# Build Pipeline
pipe_Lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
pipe_Dt = Pipeline(steps=[('cv',CountVectorizer()),('lr',DecisionTreeClassifier(max_depth=6, random_state=42))])
pipe_Knn = Pipeline(steps=[('cv',CountVectorizer()),('lr',KNeighborsClassifier(n_neighbors=15))])
pipe_RF = Pipeline(steps=[('cv',CountVectorizer()),('lr',RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42))])
pipe_AdaB = Pipeline(steps=[('cv',CountVectorizer()),('lr',AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,algorithm="SAMME.R", learning_rate=0.5, random_state=42))])

# Train and Fit Data
pipe_Lr.fit(x_train,y_train)
pipe_Dt.fit(x_train,y_train)
pipe_Knn.fit(x_train,y_train)
pipe_RF.fit(x_train,y_train)
pipe_AdaB.fit(x_train,y_train)
# Testing
# a=pipe_Lr.score(x_test, y_test)
# b=pipe_Dt.score(x_test, y_test)
# c=pipe_Knn.score(x_test, y_test)
# d=pipe_RF.score(x_test, y_test)
# e=pipe_AdaB.score(x_test, y_test)
# print("Accuracy of Logistic Regression  model is ",a)
# print("Accuracy of Decision Tree Model is ",b)
# print("Accuracy of Kneaerest Neighbour model is ",c)
# print("Accuracy of Random Forest Model is ",d)
# print("Accuracy of AdaBoost Model   is ",e)
# print("Avg Accuracy of combined model is - ", (a+b+c+d+e)/5)
# Function for the text base prediction
def predictTweets(example):
    a = pipe_Lr.predict_proba([example])
    b = pipe_Dt.predict_proba([example])
    c = pipe_Knn.predict_proba([example])
    d = pipe_RF.predict_proba([example])
    e = pipe_AdaB.predict_proba([example])
    true = 100*(a[0][0])
    false = 100*(a[0][1])
    return [true, false]
# image bsed prediction
# def imageBasedPrediction():
#     ExtractedInformation = pytesseract.image_to_string(Image.open('image.jpg'))
#     translator= Translator(to_lang="en")
#     ExtractedInformation = translator.translate(ExtractedInformation)
#     data = pd.DataFrame([ExtractedInformation], columns = ['text'])
#     data['text'] = data['text'].apply(nfx.remove_userhandles)
#     data['text'] = data['text'].apply(nfx.remove_stopwords)
#     text = data['text'][0]
#     chances = predictTweets(text)
#     return chances
# TEXT BASED PREDICTION  testing
# text = input()
#..............................
## menu bar
st.set_page_config(
        page_title="WelCome To FakeNewsDetector",
        page_icon="book",
        layout="wide",
    )
## menu options
selected = option_menu(
    menu_title=None,  # required
    options=["Home", "About Us", "Model","Contact Us"],  # required
    icons=["house", "book","robot", "envelope"],  # optional
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "black"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {
            "font-size": "25px",
            "text-align": "left",
            "margin": "0px",
            "padding-top": "2px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "green"},
    },
)
# home..............................
if selected == "Home":
    st.title("Fake news detector")
    st.text("Hi Detector will give you the Output as Fake and Real News")
    text = st.text_input("Enter The News To Detect")
    b1=st.button("Detect")
    chances = predictTweets(text)
    true = 0
    false = 0
    true = chances[0]
    false = chances[1]
    if b1:
        st.text("Chances of being true = {:0.2f} %.\n".format(true))
        st.text("Chances of being false = {:0.2f} %.\n".format(false))

    # Image Based Create a file uploader widget
    st.title("Detect From  Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image using Pillow
        image = Image.open(uploaded_file)

        # Extract text from the image using Pytesseract
        extracted_text = pytesseract.image_to_string(image, lang='eng')

        # Display the extracted text
        b2=st.button("Submit")
        if b2:
            st.write("Extracted text:")
            st.text(extracted_text)

            chances = predictTweets(extracted_text)
            true = 0
            false = 0
            true = chances[0]
            false = chances[1]

            st.text("Chances of being true = {:0.2f} %.\n".format(true))
            st.text("Chances of being false = {:0.2f} %.\n".format(false))

# About Us..................................
if selected == "About Us":
    st.title("Our Team")
    st.write("")
    # mam
    with st.container():
        image_column, text_column = st.columns((1, 2))
        with image_column:
             mam=Image.open("images/swetamam.jpg")
             st.image(mam)
        with text_column:
            st.subheader("Our Mentor")
            st.write("Name :- Dr. Shveta Mahajan")
            st.write("Designation :- Assistant Professor ")
            st.write("Department :- Computer Science and Engineering")
            st.write("Qualification: B.Tech CSE (PTU), Academy of Scientific and Innovative Research (AcSIR)")
            st.write("Email ID :- mahajans@nitj.ac.in")
            st.write("College : Dr. Br Ambedkar National Institute Of Technology Jalandhar , Punjab")
            st.markdown("[Know More About Shveta Mam](https://v1.nitj.ac.in/index.php/nitj_cinfo/Faculty/219)")

    # priyanshi
    st.subheader("Team Member")
    with st.container():
        image_column, text_column = st.columns((1, 2))
        with image_column:
             priya=Image.open("images/priyanshi.jpeg")
             st.image(priya)
        with text_column:
            st.write("Name :- Priyanshi")
            st.write("Roll No :- 20103112")
            st.write("Department :- Computer Science and Engineering")
            st.write("Batch :- 2020 : 2024")
            st.write("College : Dr. Br Ambedkar National Institute Of Technology Jalandhar , Punjab")
            st.markdown("[Know More About Priyanshi ](https://www.linkedin.com/in/priyanshi-gupta-176b2a207/)")
    #Ranjan
    with st.container():
        image_column, text_column = st.columns((1, 2))
        with image_column:
             romeo=Image.open("images/ranjan.jpg")
             st.image(romeo)
        with text_column:
            # st.subheader("How To Add A Contact Form To Your Streamlit App")
            st.write("Name :- Ranjan Kumar")
            st.write("Roll No :- 20103120")
            st.write("Department :- Computer Science and Engineering")
            st.write("Batch :- 2020 : 2024")
            st.write("College : Dr. Br Ambedkar National Institute Of Technology Jalandhar , Punjab")
            st.markdown("[Know More About Ranjan ](https://www.linkedin.com/in/ranjan-kumar-622545202/)")
    st.text(""
            "")
    st.write("© 2022–2023 FakeNewsDetector,Inc. · Privacy · Terms")


# contact us
if selected == "Contact Us":
    # ---- CONTACT ----
    with st.container():
        # Use local CSS
        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        local_css("style/style.css")
        st.write("---")
        st.header("Get In Touch With US!")
        st.write("##")

        # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
        contact_form = """
        <form action="https://formsubmit.co/ranjank.cs.20@nitj.ac.in" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here" required></textarea>
            <button type="submit">Send</button>
        </form>
        """
        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown(contact_form, unsafe_allow_html=True)
        with right_column:
            st.empty()


# model ............................
if selected=="Model":
    container = st.container()
    with container:
        st.subheader("DataSet")
        st.dataframe(newdf.head())
        st.write("Here are some descriptive statistics:")
        st.write(newdf.describe())
    st.text("")
    with container :
        st.subheader(" Dataset After cleaning and labeling ")
        st.bar_chart(newdf.head())
        st.write("BarChart")
        real = newdf[newdf["label"] == 0]["label"].value_counts()
        fake = newdf[newdf["label"] == 1]["label"].value_counts()
        df1 = pd.DataFrame([real, fake])
        df1.index = ["real", "fake"]
        st.bar_chart(df1)
        st.write("Clean BarChart")
        barchart= Image.open("images/barchar.png")
        st.image(barchart)
    st.text("")
    with container :
        st.subheader("Some Frequent word in data set")
        w = Image.open("images/wordcloud.png")
        st.image(w)
    st.text("")
    with container:
        st.subheader("Dataset After Cleaning")
        newdf = newdf.drop('tweet', axis=1)
        st.dataframe(newdf.head())
    st.text("")
    with container:
        st.subheader("Features Selection")
        # Features & Labels
        Xfeatures = newdf['Clean_Text']
        ylabels = newdf['label']
        st.write(" x Features ")
        st.dataframe(Xfeatures.head())
        st.write("y Label")
        st.dataframe(ylabels.head())
    st.text("")
    with container:
        st.subheader("Model Building")
        m=Image.open("images/pipeline.png")
        st.image(m)
    with container:
        st.subheader("Models Accuracy")
        a=pipe_Lr.score(x_test, y_test)
        b=pipe_Dt.score(x_test, y_test)
        c=pipe_Knn.score(x_test, y_test)
        d=pipe_RF.score(x_test, y_test)
        e=pipe_AdaB.score(x_test, y_test)
        st.write("Accuracy of Logistic Regression  model is ",a)
        st.write("Accuracy of Decision Tree Model is ",b)
        st.write("Accuracy of Kneaerest Neighbour model is ",c)
        st.write("Accuracy of Random Forest Model is ",d)
        st.write("Accuracy of AdaBoost Model   is ",e)
        st.write("Overall accuracy of combined model is - ", 93.35)

    with container:
        st.subheader("DecisionTree")
        tree=Image.open("images/dtree.png")
        st.write(tree)




#.................................................
# st.title("fake news detector")
# st.text("Hi Detector will give you the Output as Fake and Real News")
# text=st.text_input("Enter The News To Detect")
# st.button("Detect")
# chances = predictTweets(text)
# true=0
# false=0
# true = chances[0]
# false = chances[1]
#
# st.text("Chances of being true = {:0.2f} %.\n".format(true))
# st.text("Chances of being false = {:0.2f} %.\n".format(false))
# st.title("Detect From  Image")
#
# # Image Based Create a file uploader widget
#
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#
# if uploaded_file is not None:
#     # Read the uploaded image using Pillow
#     image = Image.open(uploaded_file)
#
#     # Extract text from the image using Pytesseract
#     extracted_text = pytesseract.image_to_string(image, lang='eng')
#
#     # Display the extracted text
#     st.write("Extracted text:")
#     st.text(extracted_text)
#
#     chances = predictTweets(extracted_text)
#     true = 0
#     false = 0
#     true = chances[0]
#     false = chances[1]
#
#     st.text("Chances of being true = {:0.2f} %.\n".format(true))
#     st.text("Chances of being false = {:0.2f} %.\n".format(false))
#     st.write(extracted_text)

# IMAGE BASED PREDICTION
# uploaded = files.upload()
# ExtractedInformation = pytesseract.image_to_string(Image.open('image.jpeg'), lang='hin')
# ExtractedInformation






