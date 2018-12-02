import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import chi2
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

#file_name = "ccrb_datatransparencyinitiative_20170207.xlsx"
#df = pd.read_excel(file_name, sheetname="Complaints_Allegations")

print df.head()
print 
print 
print df.info()


df.dropna(inplace = True)
print len(df['UniqueComplaintId'].unique())



def bar_chart(data, attribute_name):
    # This function will draw frequency bar chart for any input feature in the Dataframe
    count = data[attribute_name].value_counts()
    plt.figure(figsize=(10,10))
    sns.barplot(count.index, count.values, alpha=0.8)
    plt.xticks(rotation='vertical')
    plt.title('Number of complaints per %s'%attribute_name)
    plt.ylabel('Number of complaints', fontsize=12)
    plt.xlabel(attribute_name, fontsize=12)
    plt.show()


bar_chart(df, 'Complaint Filed Mode')
bar_chart(df, 'Complaint Filed Place')
bar_chart(df, 'Incident Location')
bar_chart(df, 'Incident Year')
bar_chart(df, 'Encounter Outcome')
bar_chart(df, 'Reason For Initial Contact')
bar_chart(df, 'Allegation Description')
bar_chart(df, 'Borough of Occurrence')


df = df[df['Borough of Occurrence'] != 'Outside NYC']

#Percentage of complaints per district
print df['Borough of Occurrence'].value_counts()*100/len(df)
print 

#Number of complaints per 100k residents in each district
df_2016 = df[df['Received Year'] == 2016]
df_2016['Borough of Occurrence'].value_counts()
population_2016 = pd.Series(data = [2629150,1455720,1643734,2333054,476015], index= ['Brooklyn', 'Bronx', 'Manhattan', 'Queens', 'Staten Island'])

print (df_2016['Borough of Occurrence'].value_counts() * 10000 / population_2016).sort_values(ascending=False)
print 

bar_chart(df, 'Received Year')
bar_chart(df, 'Close Year')

#Average year for a complaint to be closed
round((df['Close Year'] - df['Received Year']).mean(),2)

#the data for 1999-2005 and 2017 is excluded from dataframe
df = df[df['Received Year'] > 2005]
df = df[df['Received Year'] != 2017]

# Number of Stop&Frisk complaints per year
values = df.groupby('Received Year')['Complaint Contains Stop & Frisk Allegations'].sum()
print values
print 

#linear regression model for Stop&Frisk complaints per year
X = np.array(values.index).reshape(-1,1)
y = np.array(values)
lm = LinearRegression()
lm.fit(X,y)
print lm.coef_
print lm.intercept_ 

#prediction of number of the Stop&Frisk complaints in 2018
print "\npredicted value for 2018 is:", round(lm.predict(2018), 0)

#Line chart of regression model vs training data
y_predicted = lm.predict(X)
plt.plot(X, y, color = 'b', label = 'Real')
plt.plot(X, y_predicted, color ='r', label='Predicted')
plt.xlabel("Year")
plt.ylabel("Stop&Frisk Complaints")
plt.title("Number of Stop&Frinsk complaints per year")
plt.legend()
plt.show()


#is presence of a certain type of allegation indicative that a complaint will contain multiple allegations?
# building the 68467 * 4 size feature matrix X. 68467 is the number of unique complaints. 
# for each unique complaint number there is a list of size 4, in which each member corresponds 
# to a specific type of allegation

allegations_type_dict = dict()
for index, row in df.iterrows():
   allegations_type_dict[row['UniqueComplaintId']] = [0,0,0,0]

for index, row in df.iterrows():
    if row['Allegation FADO Type'] == 'Abuse of Authority':
        allegations_type_dict[row['UniqueComplaintId']][0] = 1
                        
    elif row['Allegation FADO Type'] == 'Force':
        allegations_type_dict[row['UniqueComplaintId']][1] = 1

    elif row['Allegation FADO Type'] == 'Discourtesy':
        allegations_type_dict[row['UniqueComplaintId']][2] = 1
                        
    elif row['Allegation FADO Type'] == 'Offensive Language':
        allegations_type_dict[row['UniqueComplaintId']][3] = 1

allegations_count_dict = defaultdict(int)

for index, row in df.iterrows():
    allegations_count_dict[row['UniqueComplaintId']] += 1
                          
allegations_type = []
allegations_count = []

for complain_id in allegations_type_dict:
    allegations_type.append(allegations_type_dict[complain_id])
    allegations_count.append(allegations_count_dict[complain_id])

#building a multi_variant linear regression model to find the most indicative allegaion for 
#multi-allegation complaints

X =  np.array(allegations_type)  
y =  np.array(allegations_count).reshape(-1,1)
lm = LinearRegression()
lm.fit(X,y)

print lm.coef_
print

#calculate the chi-square test statistic for testing whether a complaint is more 
#likely to receive a full investigation when it has video evidence.

X = np.array(df['Complaint Has Video Evidence']).reshape(-1,1)
y = np.array(df['Is Full Investigation']).reshape(-1,1)
print chi2(X, y)

        

#complaints_2016_df = df[df['Incident Year'] == 2016]
#complaints_2016_df = complaints_2016_df[complaints_2016_df['Borough of Occurrence'] != 'Outside NYC']
#officers =  dict(complaints_2016_df['Borough of Occurrence'].value_counts()*36000/len(complaints_2016_df))
##officers =  dict(pd.value_counts(complaints_2016_df['Borough of Occurrence'])*36000/len(complaints_2016_df))
#print officers
#precincts = {'Manhattan':22, 'Bronx':12, 'Brooklyn':23, 'Queens':16, 'Staten Island':4}
#print
#for precinct in precincts:
#    print "{} : {}".format(precinct, round(officers[precinct]/precincts[precinct], 0))

