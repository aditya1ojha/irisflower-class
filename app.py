# Importing the libraries
import streamlit as st
from keras.models import load_model 
import numpy as np 


#Loading the trained model and labels from the training file
model = load_model("model.h5")
labels = np.load("labels.npy")


# Details of the web app
st.title("IRIS FLOWER CLASSIFICATION APP")
st.caption("The Iris flower, based on its length and width of its sepal and petail, is classified into 3 types: ")
st.caption("     1. Setosa\n     2. Versicolor\n     3. Virginica")
st.caption("This app will make us determine the class which a particular Iris flower belongs to, based on the details provided by the user.")


# Asking for user input
seplen = float(st.number_input("Enter sepal length (in cm)"))
sepwid = float(st.number_input("Enter sepal width (in cm)"))
petlen = float(st.number_input("Enter petal length (in cm)"))
petwid = float(st.number_input("Enter petal width (in cm)"))

btn = st.button("PREDICT")


# Predicted result (after the button is clicked)
if btn:
	pred = model.predict(np.array([seplen, sepwid, petlen, petwid]).reshape(1,-1))
	pred = labels[np.argmax(pred)]
	st.subheader(pred)

	if pred=="Iris-setosa":
		st.image("iris_setosa.jpg")
	elif pred=="Iris-versicolor":
		st.image("iris_versicolor.jpg")
	else:	
		st.image("iris_virginica.jpg")