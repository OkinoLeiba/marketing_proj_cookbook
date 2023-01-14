# %% [markdown]
# ## <code style="background:brown;text:black"><strong>Marketing Strategy Analysis</strong>
# #### <b>Programming Script and Technical Report</b></code>

# %% [markdown]
# <a href="https://www.linkedin.c/om/in/okinoleiba" >LinkedIn: Okino Kamali Leiba</a>

# %% [markdown]
# 
# <img src="https://jupyter.org//assets/homepage/main-logo.svg" width="477" height="286" alt="jupyter logo" loading="eager" float="left">

# %% [markdown]
# 
# ### <code style="background:brown;text:black"><b>Table of Content</b></code>
# <code style="background:brown;text:black">
# <ol>
# <li>Introduction</li>
# <li>Data Loading and Quality Check</li> 
# <li>Exploratory Data Analysis</li>
# <li>Feature Additions and Engineering</li>
# <li>Statistical Analysis</li>
# <li>Final Recommendations <small>(Optimal Sales and Marketing Strategy)</small></li>
# </ol>
# </code>
# 

# %% [markdown]
# ### <code style="background:brown;text:black"><strong><b>1. Introduction</b></strong></code>

# %% [markdown]
# <code style="background:brown;text:black;">
# <ul>
# <li>What is the impact of each marketing strategy and sales visit on Sales <small>(Amount Collected)</small>?</li>
# <li>Is the same strategy valid for all the different Client Types?</li>
# </ul>
# </code>

# %% [markdown]
# <code style="background:brown;text:black;">
# Imports<br>
# <ul>
# <li>Sys module gives access to variables and functions used or maintained by the interpreter</li>
# <li>Pandas has data analysis and manipulation libraries</li>
# <li>Numpy for general array computations</li>
# <li>Matplotlib contains libraries for creating static, animated, and interactive visualizations in Python</li>
# <li>Seaborn for Python data visualization based on Matplotlib</li>
# <li>Scipy provides algorithms for scientific computing in Python</li>
# </ul></code>

# %% [markdown]
# #### <code style="background:brown;text:black;"><b>2. Data Loading and Quality Checks</b></code>

# %%

#import modules
import sys, pandas as pd, matplotlib as ml, seaborn as sns, numpy as np, scipy.stats


# %%
file_path = 'C:/Users/Owner/source/vsc_repo/marketing_prog_cookbook/campaign-data.csv'
campaign_data = pd.read_csv(file_path, sep=",", header=0, engine="c", nrows=2979, keep_default_na=True, 
encoding="utf-8")
campaign_data.columns
 

# %% [markdown]
# ### <code style="background:brown;text:black;"><b>3. Exploratory Data Analysis</b></code>
# 
# <code style="background:brown;text:black;"><b>3.1 Exploring and Understanding the Basics Data</b></code>
# <code style="background:brown;text:black;">
# <ol>
# <li>General Review and Exploration</li> 
# <li>Distribution of Data Across Different Accounts</li> 
# <li>Difference of Sales in Account Types <small>(Using Categorical Mean)</small></li> 
# <li>Statistical Summary</li> 
# </ol>
# </code>

# %%
#data exploration no visualization
#Target/Regressand/Dependent Variable: Amount Collected
#Regressor/Predictors/Indpendent Variables: Campaign (Email), Campaign (Flyer), Campaign (Phone), Sales Contact 1, Sales Contact 2,
#Sales Contact 3, Sales Contact 4, Sales Contact 5
campaign_data.head(6)


# %%

#data exploration no visualization
campaign_data.tail(6)


# %%
#data exploration no visualization
campaign_data.info()

# %%
#data exploration no visualization
#take note of Campign (Flyer) and Sales Contact 2
campaign_data[['Client ID', 'Client Type', 'Number of Customers', 'Montly Target',
       'Calendardate', 'Amount Collected', 'Unit Sold',
       'Campaign (Email)', 'Campaign (Flyer)', 'Campaign (Phone)',
       'Sales Contact 1', 'Sales Contact 2', 'Sales Contact 3',
       'Sales Contact 4', 'Sales Contact 5', 'Number of Competition']].describe().round(decimals=2)

# %%
#clean data
##modified base dataset: rename columns and changed axis: 1##
campaign_data.dropna(axis=0,how="any",)
campaign_data.duplicated(keep="first")
campaign_data.groupby('Client Type')
campaign_data = campaign_data.rename({'Montly Target':'Monthly Target','Calendardate':'Calender Date','Campaign (Email)':'Marketing Channel\
       (Email)','Campaign (Flyer)':'Marketing Channel(Flyer)', 'Campaign (Phone)':'Marketing Channel(Phone)','Number of Competition':'Level\
        of Competition'},
 axis=1,inplace=False)
#campaign_data = campaign_data.replace({'Montly Target':'Monthly Target','Calendardate':'Calender Date','Campaign (Email)':'Marketing Channel(Email)',
#'Campaign (Flyer)':'Marketing Channel(Flyer)', 'Campaign (Phone)':'Marketing Channel(Phone)','Number of Competition':'Level of Competition'}, inplace=False)
campaign_data = campaign_data.set_axis(['Client ID', 'Client Type', 'Number of Customers', 'Monthly Target',
       'Zip Code', 'Calendar Date', 'Amount Collected', 'Unit Sold',
       'Marketing Channel(Email)', 'Marketing Channel(Flyer)', 'Marketing Channel(Phone)',
       'Sales Contact 1', 'Sales Contact 2', 'Sales Contact 3',
       'Sales Contact 4', 'Sales Contact 5', 'Level of Competition'], axis=1, inplace=False)
campaign_data.head(0)



# %% [markdown]
# #### <code style="background:brown;text:black;"><b>4. Feature Additions and Engineering</b></code>

# %%
#additional date features 
##modified base dataset: added new columns##
campaign_data["Calendar Date"]=pd.to_datetime(campaign_data["Calendar Date"],errors="raise",dayfirst=True,yearfirst=True)
campaign_data["Calendar_Month"]=campaign_data["Calendar Date"].dt.month
campaign_data["Calendar_Year"]=campaign_data["Calendar Date"].dt.year



# %% [markdown]
# ### <code style="background:brown;text:black;"><b>5. Statistical Analysis</b></code>
# 
# <code style="background:brown;text:black;"><b>5.1 Statistical Analysis - Answering the Questions</b></code>
# <code style="background:brown;text:black;">
# <ol>
# <li>Impact of Marketing Strategy on Sales <small>(Using Correlation and Linear Regression)</small></li>
# <li>Impact of Competition on Sales</li>
# <li>How Different Types of Client Can Have Different Strategies <small>(Catorgize Question 1 and Question 2 Based on Account Type)</small></li>
# </ol>
# </code>

# %% [markdown]
# <code style="background:brown;text:black"><b>5.2 Impact of Marketing Strategy on Sales<b></code>

# %% [markdown]
# <code style="background:brown;text:black;">Understanding of Distrubtions</code>

# %%
campaign_data["Client Type"].value_counts(normalize=True,sort=True,ascending=True).round(decimals=2)

# %%
pd.crosstab(campaign_data["Level of Competition"], campaign_data["Client Type"],margins=True, normalize="columns").round(decimals=2)

# %%
campaign_data[['Client ID', 'Client Type', 'Number of Customers', 'Monthly Target',
       'Calendar Date', 'Amount Collected', 'Unit Sold',
       'Marketing Channel(Email)', 'Marketing Channel(Flyer)', 'Marketing Channel(Phone)',
       'Sales Contact 1', 'Sales Contact 2', 'Sales Contact 3',
       'Sales Contact 4', 'Sales Contact 5', 'Level of Competition']].groupby("Level of Competition",).mean().round(decimals=2)

# %%
campaign_data[['Client Type', 'Number of Customers', 'Monthly Target',
       'Calendar Date', 'Amount Collected', 'Unit Sold',
       'Marketing Channel(Email)', 'Marketing Channel(Flyer)', 'Marketing Channel(Phone)',
       'Sales Contact 1', 'Sales Contact 2', 'Sales Contact 3',
       'Sales Contact 4', 'Sales Contact 5']].groupby("Client Type").mean().round(2)

# %%
campaign_data[['Number of Customers', 'Monthly Target',
       'Calendar Date', 'Amount Collected', 'Unit Sold',
       'Marketing Channel(Email)', 'Marketing Channel(Flyer)', 'Marketing Channel(Phone)',
       'Sales Contact 1', 'Sales Contact 2', 'Sales Contact 3',
       'Sales Contact 4', 'Sales Contact 5']].std().round(decimals=3)

# %%
campaign_data[['Number of Customers', 'Monthly Target',
       'Calendar Date', 'Amount Collected', 'Unit Sold',
       'Marketing Channel(Email)', 'Marketing Channel(Flyer)', 'Marketing Channel(Phone)',
       'Sales Contact 1', 'Sales Contact 2', 'Sales Contact 3',
       'Sales Contact 4', 'Sales Contact 5']].var().round(decimals=3)

# %%
campaign_data.plot(x="Amount Collected", y="Marketing Channel(Email)",kind="scatter",legend=True,title="Impacting Correlation",grid=True)
campaign_data.plot(x="Amount Collected", y="Marketing Channel(Flyer)",kind="scatter",legend=True,title="Impacting Correlation",grid=True)
campaign_data.plot(x="Amount Collected", y="Marketing Channel(Phone)",kind="scatter",legend=True,title="Impacting Correlation",grid=True)

 

# %%
#strong correlation between Amount Collected and Sales Contact 2
from sklearn import preprocessing

x = np.array([element for element in campaign_data["Amount Collected"]]) #numpy arrary
y = np.array([element for element in campaign_data["Marketing Channel(Email)"]])
normalize_x = preprocessing.normalize([x] )#normalize data
normalize_y = preprocessing.normalize([y])
fig = ml.pyplot.figure() #matplotlab plot
fig, campaign_plot = ml.pyplot.subplots(figsize=(5, 2.7))
campaign_plot.scatter(normalize_x, normalize_y)
campaign_plot.set_xlabel("Amount Collected")
campaign_plot.set_ylabel("Sales Contact 2")
campaign_plot.set_title("Simple Scatter")
campaign_plot.legend(); 


# X, Y = np.meshgrid(np.linspace(-3, 3, 128), np.linspace(-3, 3, 128))
# Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
# co = campaign_plot[0,1].contourf(X, Y, Z, levels=np.linspace(-1.25, 1.25, 11))
# fig.colorbar(co, ax=campaign_plot[0, 1])




# %%
corr_data = campaign_data.corr("pearson")[["Amount Collected"]].dropna(axis=0, how="any")
corr_data.round(decimals=2)

# %% [markdown]
# <code style="background:brown;text:black;"> Correlation Analysis</code>

# %%
#consolidated strategy for targeting
import seaborn as sns, pandas as pd
correlation_data = pd.DataFrame(campaign_data[["Amount Collected","Marketing Channel(Email)","Marketing Channel(Flyer)","Marketing Channel(Phone)","Sales Contact 1",
"Sales Contact 2","Sales Contact 3","Sales Contact 4","Sales Contact 5"]].corr("pearson")["Amount Collected"]).reset_index()
correlation_data.columns = ["Impacting Variable", "Degree of Linear Impact (Correlation)"]
correlation_data = correlation_data[correlation_data["Impacting Variable"] != "Amount Collected"]
correlation_data = correlation_data.sort_values("Degree of Linear Impact (Correlation)", axis=0,ascending=False,kind="quicksort",inplace=False,
na_position="first")
correlation_data.style.background_gradient(cmap=sns.light_palette("brown",n_colors=2,reverse=False,as_cmap=True)).set_precision(2)
#correlation_data.io.Styler.background_color().set_precision(2)

# %%
import seaborn as sns, pandas as pd
correlation_data = pd.DataFrame(campaign_data.groupby("Client Type")[["Amount Collected","Marketing Channel(Email)","Marketing Channel(Flyer)","Marketing Channel(Phone)",
"Sales Contact 1","Sales Contact 2","Sales Contact 3","Sales Contact 4", "Sales Contact 5"]].corr("pearson")["Amount Collected"]).reset_index()
correlation_data = correlation_data.sort_values(["Client Type", "Amount Collected"],axis=0,ascending=False,kind="quicksort",na_position="first",inplace=False)
correlation_data.columns=["Account Type", "Variable Impact on Sales", "Impact"]
correlation_data = correlation_data[correlation_data["Variable Impact on Sales"] != "Amount Collected"].reset_index(drop=True)
correlation_data.style.background_gradient(cmap=sns.light_palette("purple",n_colors=4,reverse=False,as_cmap=True)).set_precision(2)



# %% [markdown]
# <code style="background:brown;text:black"><b>Market Strategy Impact on Sales <small>(Categorized by Different Account Type)</small></b></code>

# %%
##modified base dataset: renamed columns##
import statsmodels.api as sm, statsmodels.formula.api as smf
campaign_data.columns=[mystring.replace(" ", "_") for mystring in campaign_data.columns]
campaign_data.columns=[mystring.replace("(", "_") for mystring in campaign_data.columns]
campaign_data.columns=[mystring.replace(")", "") for mystring in campaign_data.columns]
results = smf.ols('Amount_Collected ~ Marketing_Channel_Email + Marketing_Channel_Flyer + Marketing_Channel_Phone + Sales_Contact_1 + Sales_Contact_2 + Sales_Contact_3 + Sales_Contact_4\
        + Sales_Contact_5',data=campaign_data, missing="raise", hasconst=False).fit()
print(results.summary())


# %%
html_frame = pd.read_html(results.summary().tables[1].as_html(),flavor="bs4",encoding=None,header=0,index_col=0)[0]

# %%
html_frame = html_frame.reset_index()
html_frame = html_frame[html_frame["P>|t|"] < 0.05][["index","coef"]]
#campaign_data.rename(columns={"index":"Index","coef":"Coef"})
html_frame

# %% [markdown]
# <code style="background:brown;text:black;"><b>Regression Analysis <small>(Market Sales and Strategies - Cateforized for Different Account Types)</small></b></code>

# %%
consolidated_summary=pd.DataFrame()
for acctype in list(set(list(campaign_data["Client_Type"]))):
    temp_data = campaign_data[campaign_data["Client_Type"]==acctype].copy(deep=False)
    results = smf.ols('Amount_Collected ~ Marketing_Channel_Email + Marketing_Channel_Flyer + Marketing_Channel_Phone\
         + Sales_Contact_1 + Sales_Contact_2 + Sales_Contact_3 + Sales_Contact_4\
        + Sales_Contact_5',data=temp_data, missing="raise", hasconst=False).fit()
    consolidated_frame = pd.read_html(results.summary().tables[1].as_html(),flavor="bs4",encoding=None,
    header=0,index_col=0)[0].reset_index()
    consolidated_frame = consolidated_frame[consolidated_frame["P>|t|"] < 0.05][["index","coef"]]
    consolidated_frame.columns=["Variable", "Coefficient (Impact)"]
    consolidated_frame["Account Type"] = acctype
    consolidated_frame = consolidated_frame.sort_values("Coefficient (Impact)", ascending=False, 
    kind="quicksort", inplace=False,na_position="first",axis=0)
    consolidated_frame = consolidated_frame[consolidated_frame["Variable"] != "Intercept"]
    print(acctype)
    consolidated_summary = consolidated_summary.append(consolidated_frame)
    print(consolidated_frame)
    



# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf

consolidated_summary=pd.DataFrame()
for acctype in list(set(list(campaign_data["Client_Type"]))):
    temp_data = campaign_data[campaign_data["Client_Type"]==acctype].copy(deep=False)
    results = smf.ols('Amount_Collected ~ Marketing_Channel_Email + Marketing_Channel_Flyer + Marketing_Channel_Phone\
         + Sales_Contact_1 + Sales_Contact_2 + Sales_Contact_3 + Sales_Contact_4\
        + Sales_Contact_5',data=temp_data, missing="raise", hasconst=False).fit()
    consolidated_frame = pd.read_html(results.summary().tables[1].as_html(),flavor="bs4",encoding=None,
    header=0,index_col=0)[0].reset_index()
    consolidated_frame = consolidated_frame[consolidated_frame["P>|t|"] < 0.05][["index","coef"]]
    consolidated_frame.columns=["Variable", "Coefficient (Impact)"]
    consolidated_frame["Account Type"] = acctype
    consolidated_frame = consolidated_frame.sort_values("Coefficient (Impact)", ascending=False, 
    kind="quicksort", inplace=False,na_position="first",axis=0)
    consolidated_frame = consolidated_frame[consolidated_frame["Variable"] != "Intercept"]
    print(acctype)
    consolidated_summary = consolidated_summary.append(consolidated_frame)
    print(results.summary())

# %% [markdown]
# ### <code style="background:brown;text:black">6. Final Recommendations</code>

# %% [markdown]
# <code style="background:brown;text:black">
# Using the table below we can use the coefficient to see how much return we can derive from each dollar we spend, here we can clearly see that for different Account Types and different Campaigns and different Sales Contact are effective for each type of facility<br><br>
# 
# <code style="background:brown;text:black">
# <b>Case Explanation - Small Facility</b><br>
# Small Facility achieved more impact and value on the dollar on average with Sales Contact 2. Despite the weakness in other marketing channels Small Facility is able to offset the gain in returns with Sales Contact 2. It may be advisable to conduct further inward education by developing a comprehensive marketing research plan to determine the factors contributes to the significant losses in the Marketing Channel Phone.
# </code><br><br>
# 
# <code style="background:brown;text:black">
# <b>Case Explanation - Medium Facility </b><br>
# Case Explanation - Medium Facility 
# Medium Facility shows decent results with Flyer Campaigns with each dollar spent and a return of four dollars on average. Sales Contact 2 is highly effective followed by Sales Contact 1 and Sales Contact 3. All other marketing strategies shows no significant impact on return on investment and further marketing research may be warranted to determine where improvements can be made or to determine whether to dissolve the other marketing channels.
# </code><br><br>
# 
# <code style="background:brown;text:black">
# <b>Case Explanation - Large Facility </b><br>
# Large Facility had no comparative, with the other type of facilities, and significant impact on all of its marketing channels. There is no significant data to determine whether the size of the facility or the availability of resources is a factor for return on investment based on the marketing channel. It is reasonable to assume that there is some segmentation of our target audience based on the size and type of the facility. Additional data would be necessary to determine if there is a correlation between return on investment and the segmentation of the market based on the size and type of the facility.
# </code>
# 

# %%
consolidated_summary

# %%
consolidated_summary.reset_index(inplace=True,drop=False)
consolidated_summary.drop("index",axis=1,inplace=True)
consolidated_summary["Coefficient (Impact)"] = consolidated_summary["Coefficient (Impact)"].apply(lambda x: round(x,2))
#consolidated_summary.rename({"Coefficient (Impact":"Return on Investment"})
consolidated_summary

# %%
import matplotlib

# %%
consolidated_summary.columns = ["Variable", "Return on Investment", "Account Type"]
consolidated_summary["Return on Investment"] = consolidated_summary["Return on Investment"].apply(lambda x: round(x,2))
consolidated_summary.style.background_gradient(cmap="gist_rainbow")

# %%
def format(x):
    return "${:.2f}".format(x)
consolidated_summary["Return on Investment"] = consolidated_summary["Return on Investment"].apply(format)


# %%
consolidated_summary.columns = ['Variable','Return on Investment','Account Type']
consolidated_dataframe = pd.DataFrame(consolidated_summary)
consolidated_dataframe.style.background_gradient(cmap='RdYlGn')

# consolidated_summary.columns = ["Variable", "Return on Investment", "Account Type"]
# consolidated_dataframe = pd.DataFrame(consolidated_summary)
# consolidated_dataframe.style.background_gradient(cmap="YlOrRd")
# consolidated_dataframe["Return on Investment"].style.background_gradient(cmap="gist_rainbow")



# %%
consolidated_summary.to_csv("okl_consolidated_summary.csv",mode="w",encoding="utf-8")
open("okl_consolidated_summary.csv")


