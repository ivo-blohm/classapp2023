import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

#define header of our app

st.set_page_config(page_title = "Sharky's Credit App",  page_icon=":fire:", layout="wide")


st.title("Sharky's Credit Default App")
st.markdown("This application is a *streamlit dashboard* that can be used for showing **defaulting** customers 🔥🔥")

@st.cache()
def load_data():
    data = pd.read_csv("prosper_data_app_dev.csv")
    return(data)
@st.cache()
def load_model():
    filename = "finalized_default_model.sav"
    loaded_model = pickle.load(open(filename, "rb"))
    return(loaded_model)

data = load_data()
model = load_model()

### Definition of Section 1 - Data Explorer

st.header("Customer Explorer")

row1_col1, row1_col2, row1_col3 = st.columns([1,1,1])

rate = row1_col1.slider("Interest the Customer has to pay",
                 data["borrower_rate"].min(), data["borrower_rate"].max(),
                 value=(0.10,0.15))

income = row1_col2.slider("Monthly Income of Customers",
                 0,30000, (1000,5000))

mask = ~data.columns.isin(["loan_default","borrower_rate", "employment_status"])
names = data.loc[:,mask].columns
variable = row1_col3.selectbox("Select Variables to Compare", names)


filtered_data = data.loc[(data["borrower_rate"] >= rate[0]) & 
                         (data["borrower_rate"] <= rate[1]) & 
                         (data["monthly_income"] >= income[0]) & 
                         (data["monthly_income"] <= income[1]), :]

if st.checkbox("Show filtered data", False):
    st.write(filtered_data)
    
    
row2_col1, row2_col2 = st.columns([1,1])

barplotdata = filtered_data[["loan_default", variable]].groupby("loan_default").mean()
fig1, ax = plt.subplots(figsize=(8,3.7))
ax.bar(barplotdata.index.astype(str), barplotdata[variable], color = "#fc8d62")
ax.set_ylabel(variable)

row2_col1.subheader("Compare Customer Groups")
row2_col1.pyplot(fig1, use_container_width=True)

# Definition of Section 2 - Prediction Machine 

st.header("Predicting Customer Default")
uploaded_data = st.file_uploader("Choose a file with customer data for predicting customer default")
    
if uploaded_data is not None:
    new_customers = pd.read_csv(uploaded_data)
    new_customers = pd.get_dummies(new_customers, drop_first=True)
    new_customers["predicted_default"] = model.predict(new_customers)
    
    st.success("you succesfully scored your customers")
    
    st.download_button(label = "Dowload Scored Customer Data",
                       data = new_customers.to_csv().encode("utf-8"),
                       file_name = "scored_customer_data.csv")
    
    
    
    
    
    
    
 