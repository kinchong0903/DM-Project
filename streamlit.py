import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

st.set_page_config(
    page_title="TDS 3301 - Project",
    # layout="wide",
    initial_sidebar_state="expanded",
)

st.write("""
# TDS 3301 - Project

Chong Wing Kin - 1191302246

Koh Kai Sheng - 1151104252

Lee Jun Yong - 1191302186

# (i) Exploratory Data Analysis
## - Number of times each Washer is used
""")


df = pd.read_csv("LaundryData.csv")
df = df.drop('No', axis = 1)
df.head()

grouped = df.groupby('Washer_No').size().reset_index(name= 'frequency')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
x = grouped['Washer_No'].astype(str)
y = grouped['frequency']
ax.bar(x,y)
plt.xlabel("Washer_No")
plt.ylabel("Counts")
plt.title("Number of times each Washer is used")
plt.xticks(rotation=90)
st.pyplot(fig=plt)


st.write("""
## - Number of times each Dryer is used
""")

grouped = df.groupby('Dryer_No').size().reset_index(name= 'frequency')

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])
x = grouped['Dryer_No'].astype(str)
y = grouped['frequency']
ax.bar(x,y)
plt.xlabel("Dryer_No")
plt.ylabel("Counts")
plt.title("Number of times each Dryer is used")
plt.xticks(rotation=90)
st.pyplot(fig=plt)

st.write("""
## - Number of times each race visited the shop
""")

grouped = df.groupby('Race').size().reset_index(name= 'frequency')

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])
x = grouped['Race'].astype(str)
y = grouped['frequency']
ax.bar(x,y)
plt.xlabel("Race")
plt.ylabel("Counts")
plt.title("Number of times each race visited the shop")
plt.xticks(rotation=90)
st.pyplot(fig=plt)

st.write("""
## - Number of times each basket size was used
""")

grouped = df.groupby('Basket_Size').size().reset_index(name= 'frequency')

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])
x = grouped['Basket_Size'].astype(str)
y = grouped['frequency']
ax.bar(x,y)
plt.xlabel("Basket_Size")
plt.ylabel("Counts")
plt.title("Number of times each basket size was used")
plt.xticks(rotation=90)
st.pyplot(fig=plt)

st.write("""
## - Number of times each gender visited the shop
""")

grouped = df.groupby('Gender').size().reset_index(name= 'frequency')

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])
x = grouped['Gender'].astype(str)
y = grouped['frequency']
ax.bar(x,y)
plt.xlabel("Gender")
plt.ylabel("Counts")
plt.title("Number of times each gender visited the shop")
plt.xticks(rotation=90)
st.pyplot(fig=plt)

st.write("""
## - Number of times each Body_Size visited the shop
""")

grouped = df.groupby('Body_Size').size().reset_index(name= 'frequency')

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])
x = grouped['Body_Size'].astype(str)
y = grouped['frequency']
ax.bar(x,y)
plt.xlabel("Body_Size")
plt.ylabel("Counts")
plt.title("Number of times each Body_Size visited the shop")
plt.xticks(rotation=90)
st.pyplot(fig=plt)

st.write("""
## - Number of times each customers with or without kids visited the shop
""")

grouped = df.groupby('With_Kids').size().reset_index(name= 'frequency')

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])
x = grouped['With_Kids'].astype(str)
y = grouped['frequency']
ax.bar(x,y)
plt.xlabel("With_Kids")
plt.ylabel("Counts")
plt.title("Number of times each customers with or without kids visited the shop")
plt.xticks(rotation=90)
st.pyplot(fig=plt)

st.write("""
## -  Distribution of age range of customers
""")
fig = plt.figure()
plt.hist(df['Age_Range'], bins=[0,10,20,30,40,50,60,70,80,90,100])
plt.xlabel("Age")
plt.ylabel("Counts")
plt.title("Distribution of age range of customers")
plt.xticks(rotation=90)
st.pyplot(fig=plt)

st.write("""
## -  Average sales each day of week
""")
count_number_of_days = df.groupby(['Date']).size().reset_index(name='counts')
count_number_of_days['Date'] =pd.to_datetime(count_number_of_days.Date, format='%d/%m/%Y')
count_number_of_days.sort_values(by='Date', inplace=True, ascending=True)
count_number_of_days['day_of_week'] = count_number_of_days['Date'].dt.day_name()
count_number_of_days = count_number_of_days.groupby(['day_of_week']).size().reset_index(name='num_of_days')

daily_df = df.groupby(['Date']).size().reset_index(name='counts')
daily_df['Date'] =pd.to_datetime(daily_df.Date, format='%d/%m/%Y')
daily_df.sort_values(by='Date', inplace=True, ascending=True)

daily_df['day_of_week'] = daily_df['Date'].dt.day_name()
day_of_week_df = daily_df.groupby(['day_of_week']).agg({'counts':'sum'}).reset_index()

cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_of_week_df['day_of_week'] = pd.Categorical(day_of_week_df['day_of_week'], categories=cats, ordered=True)
day_of_week_df = day_of_week_df.sort_values('day_of_week')

day_of_week_df = day_of_week_df.merge(count_number_of_days, on=['day_of_week'])

day_of_week_df['average_sales'] = round(day_of_week_df['counts'] / day_of_week_df['num_of_days'])

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])
x = day_of_week_df['day_of_week'].astype(str)
y = day_of_week_df['average_sales']
ax.bar(x,y)
plt.xlabel("Day of week")
plt.ylabel("Average sales")
plt.title("Average sales each day of week")
plt.xticks(rotation=90)
st.pyplot(fig=plt)

st.write("""
## -  Race and Basket_Size data
""")

basketRace = df[['Basket_Size','Race']]
a = basketRace.groupby(['Basket_Size','Race']).size().reset_index(name='count')
basketRaceDf = pd.DataFrame(data=a)
st.dataframe(basketRaceDf)

st.write("""
## -  Cramers V Test Results
""")

st.write("""
### (i) Is there any relationship between basket size and race?
Chi-square test showed that they are no relationship between basket size and race. Cramer's V value is not required since there is no relationship found on Chi-square test. Table above also showed they are equally distributed among the basket size.
### (ii)Is there any other relationship between basket size and other variables?
Based on chiSquare test, basket color, attire, shirt color, pants color, humidity, windspeed, wind condition are related to basket size. Furthermore, Cramer's V showed that basket colour is categorized as very strong association with basket size and rest of the variables are strong in association. The possible explanation for the strong association with basket colour should be manufacture will produce certain basket size with certain colour on market.
""")

img = Image.open("images/cramers_v_test_results.png")
st.image(img)
#################################################################################
# --------------------------- Classification Models --------------------------- #
#################################################################################

st.write("""
# Classification Models
""")
st.write("""
## - Predict customers that will choose a combination of washer and dryer
### Boruta Feature Selection
""")

df = pd.read_csv("generatedCsv/boruta_washer_dryer.csv")
st.dataframe(df)

st.markdown(
    """
### Classification Models Result

| Classifier | Score |
| --- | --- |
| KNeighborsClassifier | 0.084 |
| Naive Bayes | 0.119 |
""")

st.write("""
## - Predict customers that will choose a washer number
### Boruta Feature Selection
""")

df = pd.read_csv("generatedCsv/boruta_washer.csv")
st.dataframe(df)

st.markdown(
    """
### Classification Models Result

| Classifier | Score |
| --- | --- |
| KNeighborsClassifier | 0.342 |
| Naive Bayes | 0.322 |
""")

st.write("""
## - Predict customers that will choose a dryer number
### Boruta Feature Selection
""")

df = pd.read_csv("generatedCsv/boruta_dryer.csv")
st.dataframe(df)

st.markdown(
    """
### Classification Models Result

| Classifier | Score |
| --- | --- |
| KNeighborsClassifier | 0.252 |
| Naive Bayes | 0.287 |
""")

st.write("""
## - Predict customer's basket size
### Using all features
""")
st.markdown(
    """
### Classification Models Result For Basket Size

| Classifier | Accurancy | AUC | Precision |
| --- | --- |
| KNeighborsClassifier | 0.851 | 0.90 | 0.81 |
| Naive Bayes | 0.733 | 0.84 | 0.80 |
| Random Forest | 0.944 | 0.99 | 0.96 |
| Logistic Regression | 0.770 | 0.82 | 0.80 |

""")
st.markdown("""
### ROC curve on all features 
""")
imgROC1 = Image.open("images/BasketSizeROC1.png")
st.image(imgROC1)

st.write("""
### Using selected features (Basket_Size, Basket_colour, Attire, Shirt_Colour, Pants_Colour, Humidity, Wind Speed, Condition)
""")
st.markdown(
    """
### Classification Models Result For Basket Size

| Classifier | Accurancy | AUC | Precision |
| --- | --- |
| KNeighborsClassifier | 0.95 | 0.90 | 0.86 |
| Naive Bayes | 0.740 | 0.78 | 0.80 |
| Random Forest | 0.934 | 0.98 | 0.95 |
| Logistic Regression | 0.710 | 0.74 | 0.77 |

""")

st.markdown("""
### ROC curve on selected features 
""")
imgROC2 = Image.open("images/BasketSizeROC2.png")
st.image(imgROC2)





#################################################################################
# -------------------------- Association Rule Mining -------------------------- #
#################################################################################

st.write("""
# Association rule mining
### Question: What combination of washer and dryer does customer used frequently
""")

st.write("""
---

(Rule 1) Dryer_No 10 -> Washer_No 6

Support: 0.084

Confidence: 0.3436

Lift: 1.3295

---

(Rule 2) Dryer_No 7 -> Washer_No 3

Support: 0.11

Confidence: 0.381

Lift: 1.3562

---
(Rule 3) Dryer_No 8 -> Washer_No 4

Support: 0.076

Confidence: 0.3112

Lift: 1.3697

---
(Rule 4) Washer_No 5 -> Dryer_No 9

Support: 0.069

Confidence: 0.3073

Lift: 1.3161

---
""")

#################################################################################
# --------------------------------- Regressions ------------------------------- #
#################################################################################

st.markdown(
    """
## Predicting number of sales of a day based on the number of sales on previous day using ARIMA, RNN and LSTM

Below shows the number of sales each day of 23 days.
""")

option = st.selectbox("Select type of graph", ['Line', 'Bar'])

if option == 'Line':
    img = Image.open("images/daily_sales.png")
    st.image(img)
elif option == 'Bar':
    img = Image.open("images/daily_sales_bar.png")
    st.image(img)

arima = "Parameters used for ARIMA are p = 1, d = 1 and q = 1."
rnn = "RNN model uses single layer simpleRNN with 128 units, added with a dropout layer with rate = 0.3 and last layer is a dense layer with 1 unit. Optimizer used is 'adam' and loss function used is 'mean square error'."
lstm = "LSTM model uses two LSTM layer with 128 units, a dropout layer is then added with rate = 0.2. Lastly, a dense layer with 1 unit is added as the output layer. Optimizer used is 'adam' and loss function used is 'mean square error'."

st.write(
"""


70% of the data is used for training and 30% is for testing.

Results of predicting on test set using ARIMA, RNN and LSTM are shown below.

""")
option = st.selectbox("Select models", ['ARIMA', 'RNN', 'LSTM'])

if option == 'ARIMA':
    img = Image.open("images/arima.png")
    st.write(arima)
    st.image(img)
elif option == 'RNN':
    img = Image.open("images/rnn.png")
    st.write(rnn)
    st.image(img)
elif option == 'LSTM':
    img = Image.open("images/lstm.png")
    st.write(lstm)
    st.image(img)

#################################################################################
# ---------------------------------  ------------------------------- #
#################################################################################