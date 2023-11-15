import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Define Load functions
@st.cache_data
def load_data():
    data = pd.read_csv("prosper_data_app_dev.csv")
    return(data.dropna())

@st.cache_resource
def load_model():
    filename = "finalized_default_model.sav"
    loaded_model = pickle.load(open(filename, "rb"))
    return(loaded_model)

# Load Data
data = load_data()
model = load_model()

#### Define Header of app
#########################################################

st.title("Sharky's Credit Default App")
st.markdown("🤑💥💰 This application is a Streamlit dashboard that can be used to *analyze* and **predict** credit default 🤑💥💰")


#### Definition of Section 1 for exploring data
#########################################################

st.header("Customer Explorer")

# Introducing three colums for user inputs
row1_col1, row1_col2, row1_col3 = st.columns([1,1,1])

rate = row1_col1.slider("Interest the customer has to pay",
                  data["borrower_rate"].min(),
                  data["borrower_rate"].max(),
                  (0.05, 0.15))

income = row1_col2.slider("Monthly Income of Customers",
                  data["monthly_income"].min(),
                  data["monthly_income"].max(),
                  (2000.00, 30000.00))

mask = ~data.columns.isin(["loan_default","employment_status"])
names = data.loc[:,mask].columns
variable = row1_col3.selectbox("Select Variable to Compare", names)

# creating filtered data set according to slider inputs
filtered_data = data.loc[(data["borrower_rate"] >= rate[0]) & 
                         (data["borrower_rate"] <= rate[1]) &
                         (data["monthly_income"] >= income[0]) & 
                         (data["monthly_income"] <= income[1]), : ]

# Add checkbox allowing us to display raw data
if st.checkbox("Show Filtered Data", False):
    st.subheader("Raw Data")
    st.table(filtered_data)
    
row2_col1, row2_col2 = st.columns([1,1])

barplotdata = filtered_data[["loan_default",variable]].groupby("loan_default").mean()
fig1, ax = plt.subplots(figsize=(8,3.7))
ax.bar(barplotdata.index.astype(str), barplotdata[variable], color="#fc8d62")

row2_col1.subheader("Compare Customer Groups")
row2_col1.pyplot(fig1, use_container_width=True)

row2_col2.markdown("plot2")



st.header("Predicting Customer Support")

uploaded_data = st.file_uploader("Choose a file with Customer Data for Predicting Customer Default")

if uploaded_data is not None:
    new_customers = pd.read_csv(uploaded_data)
    new_customers = pd.get_dummies(new_customers, drop_first=True)
    new_customers["Prediction"] = model.predict(new_customers)
    
    st.success("you succesfully scored your customers")
    
    st.download_button(label="Download Scored Customer Data",
                data = new_customers.to_csv().encode("utf-8"),
                file_name="scored_customer_data.csv")
    
    