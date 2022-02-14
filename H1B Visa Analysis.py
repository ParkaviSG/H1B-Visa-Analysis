import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from mpl_toolkits.basemap import Basemap
from decimal import Decimal
import matplotlib as mpl
from matplotlib.pyplot import pie,axis,show
import tkinter
from matplotlib import style
plt.style.use('seaborn')
from sklearn.cluster import KMeans
import time
import scipy as sci

root = tkinter.Tk()
frame = tkinter.Frame(root)
frame.pack()


def datasample():
    
    f = pd.read_csv("h1b_kaggle.csv")
    del f['Unnamed: 0']

    #to remove the missing characters and index in the data set

    f = f.dropna()
    f.reset_index()
    lng = len(f)
    
    
    print("NUMBER OF DATA SAMPLES INITIALLY PRESENT                       : ",len(f))
    print("NUMBER OF DATA SAMPLES AFTER THE REMOVAL OF MISSING CHARACTERS : ",lng)


def topcompanies():
    
    f = pd.read_csv("h1b_kaggle.csv")
    del f['Unnamed: 0']

    #to remove the missing characters and index in the data set

    f = f.dropna()
    f.reset_index()
    lng = len(f)
    
    
    #list of top 15 employing companies

    print("LIST OF TOP 15 EMPLOYING COMPANIES : \n")
    print(f.EMPLOYER_NAME.value_counts().head(15))
    
    #visualising the top 15 employing companies - bar graph
    
    fig = f["EMPLOYER_NAME"].value_counts().head(15).plot(kind = 'barh',color='black',title = 'TOP 15 HIRING COMPANIES')
    fig.set_facecolor("lightgreen")
    show()

    #visualising the top 15 employing companies - lollipop graph
    values=f["EMPLOYER_NAME"].value_counts().head(15)
    plt.stem(values, markerfmt=' ')
    (markers, stemlines, baseline) = plt.stem(values)
    plt.setp(markers, marker='*', markersize=10, markeredgecolor="goldenrod", markeredgewidth=2)
    plt.title("TOP 15 HIRING COMPANIES")
    show()


    #xticks
    
    values = f["EMPLOYER_NAME"].head(15)
    counts = f["EMPLOYER_NAME"].value_counts().head(15)

    plt.figure(1)
    x = range(15)
    plt.xticks(x, values,rotation=90)
    plt.plot(x,counts,"g")

    plt.show()


def wageanalysis():
    
    f = pd.read_csv("h1b_kaggle.csv")
    del f['Unnamed: 0']

    #to remove the missing characters and index in the data set

    f = f.dropna()
    f.reset_index()
    lng = len(f)
    
    
    
    #analysing the wages

    print("ANALYSING THE WAGES : \n")
    print(f.PREVAILING_WAGE.value_counts().sort_values(ascending=False).head())

    #calculating the mean wage

    print("MEAN WAGE : ",f.PREVAILING_WAGE.mean())

    #visualising the wages given by the employers - bar

    fig1 = f.groupby(['EMPLOYER_NAME']).mean()['PREVAILING_WAGE'].nlargest(15).plot(kind='bar',color='teal',title='WAGES GIVEN BY EMPLOYEES')
    fig1.set_facecolor("k")
    show()

    #visualising the wages given by the employers - lallipop graph

    values=f.groupby(['EMPLOYER_NAME']).mean()['PREVAILING_WAGE'].nlargest(15)
    plt.stem(values, markerfmt=' ')
    (markers, stemlines, baseline) = plt.stem(values)
    plt.setp(markers, marker='o', markersize=10, markeredgecolor="mediumvioletred", markeredgewidth=1)
    plt.title("WAGES BY EMPLOYEES")
    show()



def maximumjobs():
    
    f = pd.read_csv("h1b_kaggle.csv")
    del f['Unnamed: 0']

    #to remove the missing characters and index in the data set

    f = f.dropna()
    f.reset_index()
    lng = len(f)
    
    
    #finding cities with maximum jobs

    print("CITIES WHERE YOU CAN FIND MAXIMUM JOBS : \n")
    print(f['WORKSITE'].value_counts().head(20))
    print("\n")
    
    #visualising the cities where one can find maximum jobs - bar graph

    fig2 = f["WORKSITE"].value_counts().head(20).plot(kind = 'bar',color='maroon',title='CITIES WITH MAXIMUM JOBS')
    fig2.set_facecolor("lavenderblush")
    show()
    
    #xticks
    
    values = f["WORKSITE"].head(20)
    counts = f["WORKSITE"].value_counts().head(20)

    plt.figure(1)
    x = range(20)
    plt.xticks(x, values,rotation=90)
    plt.plot(x,counts,"g")

    plt.show()

    #visualising the cities where one can find maximum jobs - lallipop graph

    values=f["WORKSITE"].value_counts().head(20)
    plt.stem(values, markerfmt=' ')
    (markers, stemlines, baseline) = plt.stem(values)
    plt.setp(markers, marker='h', markersize=10, markeredgecolor="lime", markeredgewidth=2)
    plt.title("CITIES WITH MAXIMUM JOBS")
    show()

def petitionanalysis():
    
    f = pd.read_csv("h1b_kaggle.csv")
    del f['Unnamed: 0']

    #to remove the missing characters and index in the data set

    f = f.dropna()
    f.reset_index()
    lng = len(f)
    
    
    
    #calculating the number of states
    
    f.loc[:,'WORKSITE'] = f.loc[:,'WORKSITE'].apply(lambda rec:rec.split(',')[1][1:])

    def change_NA(rec):
        if(rec=='NA'):
            return 'MARINA ISLANDS'
        return rec

    f.loc[:,'WORKSITE'] = f.loc[:,'WORKSITE'].apply(lambda rec: change_NA(rec))
    
    #renaming column headings

    f.rename(columns={'EMPLOYER_NAME': 'EMPLOYER','FULL_TIME_POSITION': 'FULL_T','PREVAILING_WAGE': 'PREV_WAGE','WORKSITE': 'STATE','lon': 'LON','lat': 'LAT'}, inplace=True)
    columns_to_keep=['CASE_STATUS','YEAR','STATE','SOC_NAME','JOB_TITLE','FULL_T','PREV_WAGE','EMPLOYER','LON','LAT']
    f = f[columns_to_keep]
    f.columns

    f['LON'] = f['LON'].apply(lambda lon:float("%.2f" %lon))
    f['LAT'] = f['LAT'].apply(lambda lat:float("%.2f" %lat))
    f['YEAR'] = f['YEAR'].apply(lambda year:'%g' %(Decimal(str(year))))
    f['PREV_WAGE'] = f['PREV_WAGE'].apply(lambda wage:'%g' %(Decimal(str(wage))))
    print(f.head(2))


    f['CASE_STATUS'].unique()
    #calculating the petitions distributions by status

    status_freq=[0]*7
    statuses=['CERTIFIED-WITHDRAWN','CERTIFIED','DENIED','WITHDRAWN','REJECTED','INVALIDATED','PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED']
    for i in range(0,7):
        status_freq[i] = f[f.CASE_STATUS==statuses[i]]['CASE_STATUS'].count()
    status_freq
    plt.figure(figsize=(4.5,4.5))
    plt.title('PETITIONS BY CASE STATUS')
    axis('equal')
    pie(status_freq[:4],labels=statuses[:4],autopct = '%1.0f%%')
    show()

    #calculating petioins distributed in years

    years = ['2011','2012','2013','2014','2015','2016']
    year_count = [0]*6
    for i in range(0,6):
        year_count[i] = f[f.YEAR==years[i]]['YEAR'].count()
    print(year_count)
    
    #scatter plot
    
    df=pd.DataFrame({'x': years, 'y': year_count})#, 'group': np.repeat('A',20000) })
    plt.plot( 'x', 'y', data=df, linestyle='', marker='o')
    plt.xlabel('YEARS')
    plt.ylabel('NUMBER OF PETITIONS')
    plt.title('PETITION DISTRIBUTION BY YEARS')
    
    plt.show()
    
    
    #plotting the petitions distributions by years

    sn.set_context("notebook",font_scale=1.0)
    plt.figure(figsize=(13,3))
    plt.title('PETITIONS DISTRIBUTION BY YEAR')
    sn.countplot(f['YEAR'])


    #plotting the petitions distributions by years - pie


    plt.figure(figsize=(4.5,4.5))
    plt.title('PETITIONS DISTRIBUTION BY YEARS')
    axis('equal')
    pie(year_count[:6],labels=years[:6],autopct = '%1.0f%%')
    show()

    #plotting the petitions distributions by years - donut

    my_circle = plt.Circle((0,0), 0.7, color='white')

    plt.figure(figsize=(4.5,4.5))
    plt.title('PETITIONS DISTRIBUTION BY YEARS')
    axis('equal')
    plt.pie(year_count[:6], labels=years[:6], autopct='%1.0f%%')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.show()

    
    #analysing denied petitions

    denied = f[f.CASE_STATUS=='DENIED']
    d = len(denied)
    print("The number of denied petitions is : ",d)

    #denied petitions in a tabular form

    del denied['CASE_STATUS']
    denied = denied.reset_index()
    denied.head(2)
    show()

    #calculating denied distributions by year

    denied_year_count = [0]*6
    for i in range(0,6):
        denied_year_count[i] = denied[denied.YEAR==years[i]]['YEAR'].count()
    print(denied_year_count)

    #plotting the denied petitions by year

    sn.set_context("notebook",font_scale=1.0)
    plt.figure(figsize=(13,3))
    plt.title('DENIED PETITIONS DISTRIBUTION BY YEAR')
    sn.countplot(denied['YEAR'])

    #denied petitions in the form of pie chart
    plt.figure(figsize=(4.5,4.5))
    plt.title('DENIED PETITIONS BY YEAR')
    axis('equal')
    pie(denied_year_count[:6],labels=years[:6],autopct='%1.0f%%')
    show()
    #denied petitions - donut

    my_circle = plt.Circle((0,0), 0.7, color='white')

    plt.figure(figsize=(4.5,4.5))
    plt.title('DENIED PETITIONS BY YEARS')
    axis('equal')
    plt.pie(denied_year_count[:6], labels=years[:6], autopct='%1.0f%%')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.show()

    #calculating the rate at which the petitions are denied per year

    denied_year_rate = [0]*6
    for i in range(0,6):
        denied_year_rate[i] = float("%.2f" % ((denied_year_count[i]/year_count[i])*100))
    ratio = pd.DataFrame()
    ratio['year'] = years
    ratio['denied rate %'] = denied_year_rate
    ratio = ratio.set_index(['year'])
    ratio.T
    show()

    #plotting the denied rate by year in a bar graph

    ratio = ratio.reset_index()
    sn.set_context('notebook',font_scale=1.0)
    plt.figure(figsize=(13,3))
    plt.title("DENIED PETITIONS RATE BY YEAR")
    g = sn.barplot(x='year',y='denied rate %',data=ratio)
    #plotting the denied rate by year in pie chart
    plt.figure(figsize=(4.5,4.5))
    plt.title('DENIED RATE BY YEAR')
    axis('equal')
    pie(denied_year_rate[:6],labels=years[:6])
    show()
    #plotting the denied rate by year in donut chart

    my_circle = plt.Circle((0,0), 0.7, color='white')

    plt.figure(figsize=(4.5,4.5))
    plt.title('DENIED RATE BY YEARS')
    axis('equal')
    plt.pie(denied_year_rate[:6], labels=years[:6], autopct='%1.0f%%')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.show()

    #calculating the number of petitions filled by the state

    US_states = ['ALABANA','ALASKA','ARIZONA','ARKANSAS','CALIFORNIA','COLORADO','CONNECTICUT','DELAWARE','DISTRICT OF COLOMBIA','FLORIDA','GEORGIA','HAWAII','IDAHO','IOWA','KANSAS','KENTUCKY','LOUISIANA','MAINE','MARIANA ISLANDS','MARYLAND','MASSACHUSETTS','MICHIGAN','MINNESOTA','MISSISSIPPI','MISSOURI','MONTANA','NEBRASKA','NEVEDA','NEW HAMPSHIRE','NEW JERSEY','NEW MEXICO','NEW YORK','NORTH CAROLINA','NORTH DAKOTA','OHIO','OKLAHOMA','OREGON','PENNSYLVANIA','PEURTO RICO','RHODE ISLAND','SOUTH CAROLIN','SOUTH DAKOTA','TENNESSE','TEXAS','UTAH','VERMONT','VIRGINIA','WASHINGTON','WEST VIRGINIA','WISCONSIN','WYOMING']

    petitions_by_state = [0]*51
    for i in range(0,51):
        petitions_by_state[i] = f[f.STATE==US_states[i]]['STATE'].count()
    pet_state = pd.DataFrame()
    pet_state['STATE'] = US_states
    pet_state['FILED PETITIONS'] = petitions_by_state
    print(sum(petitions_by_state))

    #number of petitions filed by the state in pie chart

    plt.figure(figsize=(4.5,4.5))
    plt.title('NUMBER OF PETITIONS FILED BY EACH STATE')
    axis('equal')
    c =["black","bisque","forestgreen","slategrey","darkorange","lighsteelblue","burlywood","antiquewhite","tan","lavender","mediumspringgreen","orange","wheat","darkgoldenrod","navy","firebrick","khaki","indigo","darkred","teal","olive","tomato","deeppink","orchid","palegreen","peru","yellow","saddlebrown","pink","cadetblue","aqua","lightyellow","magenta","palevioletred","plum","azure","maroon","gainsboro","lightcoral","honeydew","seagreen","papayawhip","royalblue","cornsilk","blueviolet","olivedrab","mistyrose","salmon","navajowhite","ghostwhite","paleturquoise","powderblue","lime","beige","deepskyblue","hotpink"]
    pie(petitions_by_state[:53])
    plt.legend(labels=US_states[:53], loc='center right', bbox_to_anchor=(-0.1, 1.),fontsize=8)
    plt.tight_layout()
    show()

    #number of petitions filed by the state in donut chart

    my_circle = plt.Circle((0,0), 0.7, color='plum')

    plt.figure(figsize=(4.5,4.5))
    plt.title('NUMBER OF PETITIONS FILED BY EACH STATE')
    axis('equal')

    plt.pie(petitions_by_state[:53])
    plt.legend(labels=US_states[:53], loc='center right', bbox_to_anchor=(-0.1, 1.),fontsize=8)
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.show()
    #analysing filed petions by state

    sn.set_context("notebook",font_scale=1.0)
    plt.figure(figsize=(13,5))
    plt.title("FILED PETITIONS BY STATE")
    v = sn.barplot(x='STATE',y='FILED PETITIONS',data=pet_state)
    rotg = v.set_xticklabels(v.get_xticklabels(),rotation=90)

    #calculating number of petitions denied by the state

    denied_by_state = [0]*51
    for i in range(0,51):
        denied_by_state[i] = denied[denied.STATE==US_states[i]]['STATE'].count()
    den_state = pd.DataFrame()
    den_state['STATE'] = US_states
    den_state['DENIED PETITIONS'] = denied_by_state
    print(sum(denied_by_state))

    #plotting the number of petitions denied by the state

    sn.set_context("notebook",font_scale=1.0)
    plt.figure(figsize=(13,5))
    plt.title("DENIED PETITIONS BY STATE")
    v = sn.barplot(x='STATE',y='DENIED PETITIONS',data=den_state)
    rotg = v.set_xticklabels(v.get_xticklabels(),rotation=90)

    #plotting the number of petitions denied by the state in pie chart

    plt.figure(figsize=(4.5,4.5))
    plt.title('PETITIONS DENIED BY EACH STATE')
    axis('equal')
    c =["black","bisque","forestgreen","slategrey","darkorange","lighsteelblue","burlywood","antiquewhite","tan","lavender","mediumspringgreen","orange","wheat","darkgoldenrod","navy","firebrick","khaki","indigo","darkred","teal","olive","tomato","deeppink","orchid","palegreen","peru","yellow","saddlebrown","pink","cadetblue","aqua","lightyellow","magenta","palevioletred","plum","azure","maroon","gainsboro","lightcoral","honeydew","seagreen","papayawhip","royalblue","cornsilk","blueviolet","olivedrab","mistyrose","salmon","navajowhite","ghostwhite","paleturquoise","powderblue","lime","beige","deepskyblue","hotpink"]
    pie(denied_by_state[:53])
    plt.legend(labels=US_states[:53], loc='center right', bbox_to_anchor=(-0.1, 1.),fontsize=8)
    plt.tight_layout()
    show()


    #plotting the number of petitions denied by the state in donut chart


    my_circle = plt.Circle((0,0), 0.7, color='plum')

    plt.figure(figsize=(4.5,4.5))
    plt.title('PETITIONS DENED BY EACH STATE')
    axis('equal')

    plt.pie(denied_by_state[:53])
    plt.legend(labels=US_states[:53], loc='center right', bbox_to_anchor=(-0.1, 1.),fontsize=8)
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.show()

    #rate at which the petitions are denied
    denied_state_rate = [0]*51
    for i in range(0,51):
        denied_state_rate[i]=float("%.2f" %((denied_by_state[i]/petitions_by_state[i])*100))
    ratios = pd.DataFrame()
    ratios['STATE']=US_states
    ratios['DENIED PETITIONS %'] = denied_state_rate

    #visualising bar graph
    sn.set_context("notebook",font_scale=1.0)
    plt.figure(figsize=(13,5))
    plt.title("DENIED PETITIONS RATE IN % BY STATES")
    v = sn.barplot(x='STATE', y='DENIED PETITIONS %',data=ratios)
    rotg=v.set_xticklabels(v.get_xticklabels(),rotation=90)


    #tabulating whether the petitions are filed/denied along with denied petitions rate

    pet_state['DENIED PETITIONS'] = denied_by_state
    pet_state['DENIED PETITIONS %'] = denied_state_rate
    pet_state = pet_state.sort_values(by='DENIED PETITIONS %',ascending= False)
    pet_state
    show() 


def topjobtitles():
    
    f = pd.read_csv("h1b_kaggle.csv")
    del f['Unnamed: 0']

    #to remove the missing characters and index in the data set

    f = f.dropna()
    f.reset_index()
    lng = len(f)
    
    #top 25 job titles

    print(f.JOB_TITLE.value_counts().sort_values(ascending=False).head(25))

    #plotting the top 25 jobs

    fig3 = f.JOB_TITLE.value_counts().sort_values(ascending=False).head(25).plot(kind='barh',color='lightsalmon')
    fig3.set_facecolor("darkolivegreen")
    plt.title("TOP 25 JOB TITLES")
    show()


    #plotting in lollipop graph

    values=f.JOB_TITLE.value_counts().sort_values(ascending=False).head(25)
    plt.stem(values, markerfmt=' ')
    (markers, stemlines, baseline) = plt.stem(values)
    plt.setp(markers, marker='8', markersize=10, markeredgecolor="aqua", markeredgewidth=2)
    plt.title("TOP 25 JOB TITLES")
    show()


def extra():

    df=pd.read_csv("h1b_kaggle.csv")
    df2=df[["EMPLOYER_NAME","PREVAILING_WAGE", "YEAR", "WORKSITE", "lon", "lat"]].dropna(axis=0, how='any')
    df2=df2[df2["PREVAILING_WAGE"]<250000]
    df2=df2[df2["PREVAILING_WAGE"]>1]
    df2["PREVAILING_WAGE"]=df2["PREVAILING_WAGE"]/1000


    #Number of petitions based on years
    df2["YEAR"]=df2["YEAR"].astype(int)
    year=df2["YEAR"].value_counts(ascending=True)
    year.plot.bar(color='r')
    plt.xlabel('Year', size=15)
    plt.ylabel('Counts', size=15)
    plt.title('Number of petitions by year',size=20)
    plt.xticks(rotation=0)
    plt.show()

    #plot histogram
    df2["PREVAILING_WAGE"].hist(bins=50, color='r',rwidth=0.8)
    plt.xlabel('Salary (k)', size=15)
    plt.ylabel('Counts', size=15)
    plt.title('Histogram of H1B wage',size=20)
    average=int(df2["PREVAILING_WAGE"].mean())
    median=int(df2["PREVAILING_WAGE"].median())
    plt.text(175, 310000, 'Average=%sk\n Median=%sk' %(average,median),size=15)
    plt.grid(True)
    plt.show()
   
    ave_wage=df2.pivot_table(index="YEAR", values="PREVAILING_WAGE")
    plt.plot(ave_wage, '-', color='r')
    plt.plot(ave_wage,'o',color='b')
    plt.ylabel('Average salary in k', size=15)
    plt.xlabel('year', size=15)
    plt.show()
    
    # How the salary distribution has evolved with respect to year?
    sn.violinplot(x="YEAR", y="PREVAILING_WAGE",data=df2)
    plt.ylabel('salary in k', size=15)
    plt.xlabel('year', size=15)
    plt.ylim(-10,150)
    plt.show()
"""
    # setup Lambert Conformal basemap.
    m = Basemap(llcrnrlon=-119,llcrnrlat=20,urcrnrlon=-64,urcrnrlat=49,projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    x,y=m(highwage.lon.tolist(), highwage.lat.tolist())
    plt.scatter(x, y, 2, color='r')
    m.drawcoastlines()
    m.drawmapboundary(fill_color='white')
    m.drawstates()
    m.drawcountries(linewidth=2)
    plt.title('Locations of highly paid jobs',size=15)
    plt.show()

"""

def corelation():
    H1Info = pd.read_csv('h1b_kaggle.csv')

    columnNames = H1Info.columns.values
    numberNan = np.zeros(len(columnNames),dtype=np.uint32)
    i = 0
    for k in columnNames:
        numberNan[i] = H1Info[k].isnull().sum()
        i += 1
        
    H1City,H1State = H1Info['WORKSITE'].str.split(', ',1).str #Get the states values
        
    #Putting everything in a dataframe
    H1StateCounts = H1State.value_counts()
    H1StateCounts = H1StateCounts.to_frame()
    H1StateCounts.columns = ['People']
    H1StateCounts = H1StateCounts[H1StateCounts.index!='NA']
    H1StateCounts = H1StateCounts.sort_index()
        
    #Importing population data from the 2010 census and merging it with the dataframe with the H1 values
    PopInfo = [4779736, 710231, 6392017, 2915918, 37253956, 5029196,
               3574097, 897934, 601723, 18801310, 9687653, 1360301, 
               1567582, 12830632, 6483802, 3046355, 2853118, 4339367, 
               4533372, 1328361, 5773552, 6547629, 9883640, 5303925, 
               2967297, 5988927, 989415, 1826341, 2700551, 1316470, 8791894, 
               2059179, 19378102, 9535483, 672591, 11536504, 3751351, 3831074, 
               12702379, 3725789, 1052567, 4625364, 814180, 6346105, 25145561, 
               2763885, 625741, 8001024, 6724540, 1852994, 5686986, 563626]
    H1StateCounts['TotalPop'] = PopInfo
    
    #Compute the H1s per capita
    H1StateCounts['H1PerCapita'] = np.divide(H1StateCounts['People'].values,H1StateCounts['TotalPop'].values)
    
    #Winning party in the 2016 election (double letter means >10 pp. victory difference)
    H1StateCounts['Vote'] = ['RR','RR','R','RR','DD','D','DD','DD','DD','R','R','DD','RR','DD','RR','R','RR','RR','RR','D','DD','DD','R','D','RR','RR','RR','RR','D','D','DD','D','DD','R','RR','R','RR','DD','R','N','DD','RR','RR','RR','R','RR','DD','D','DD','RR','R','RR']

    #Sort them by H1s per Capita
    p = H1StateCounts.sort_values('H1PerCapita',ascending=False)
    print(p)
    
    #Reading Latitude and Longitude data (taking out N/As)
    H1LatLong = H1Info.loc[H1State != 'NA']
    H1LatLong = H1LatLong[['lon','lat']]
    H1LatLong = H1LatLong.dropna();
    H1Long = H1LatLong['lon'].values
    H1Lat = H1LatLong['lat'].values
    #K-Means clustering to see where the applicants were
    nClusters = 10
    kmeans = KMeans(n_clusters=nClusters, random_state=0).fit(H1LatLong.values)

    kl = kmeans.labels_ #Getting the cluster for each case
    H1LatLong['Cluster'] = kl

    H1LatLong['State'] = H1State
    H1LatLong = H1LatLong.dropna()
    H1LatLong.sort_values('Cluster')
    print(H1LatLong[['Cluster','State']].drop_duplicates().sort_values('Cluster')) #See which states a cluster cover


    pltIndex = range(0,10501,5)

    plt.plot(H1LatLong.iloc[pltIndex,0].loc[H1LatLong.iloc[pltIndex,2] == 0], H1LatLong.iloc[pltIndex,1].loc[H1LatLong.iloc[pltIndex,2] == 0], "ro")
    plt.plot(H1LatLong.iloc[pltIndex,0].loc[H1LatLong.iloc[pltIndex,2] == 1], H1LatLong.iloc[pltIndex,1].loc[H1LatLong.iloc[pltIndex,2] == 1], "go")
    plt.plot(H1LatLong.iloc[pltIndex,0].loc[H1LatLong.iloc[pltIndex,2] == 2], H1LatLong.iloc[pltIndex,1].loc[H1LatLong.iloc[pltIndex,2] == 2], "bo")
    plt.plot(H1LatLong.iloc[pltIndex,0].loc[H1LatLong.iloc[pltIndex,2] == 3], H1LatLong.iloc[pltIndex,1].loc[H1LatLong.iloc[pltIndex,2] == 3], "co")
    plt.plot(H1LatLong.iloc[pltIndex,0].loc[H1LatLong.iloc[pltIndex,2] == 4], H1LatLong.iloc[pltIndex,1].loc[H1LatLong.iloc[pltIndex,2] == 4], "mo")
    plt.plot(H1LatLong.iloc[pltIndex,0].loc[H1LatLong.iloc[pltIndex,2] == 5], H1LatLong.iloc[pltIndex,1].loc[H1LatLong.iloc[pltIndex,2] == 5], "ko")
    plt.plot(H1LatLong.iloc[pltIndex,0].loc[H1LatLong.iloc[pltIndex,2] == 6], H1LatLong.iloc[pltIndex,1].loc[H1LatLong.iloc[pltIndex,2] == 6], "r*")
    plt.plot(H1LatLong.iloc[pltIndex,0].loc[H1LatLong.iloc[pltIndex,2] == 7], H1LatLong.iloc[pltIndex,1].loc[H1LatLong.iloc[pltIndex,2] == 7], "g*")
    plt.plot(H1LatLong.iloc[pltIndex,0].loc[H1LatLong.iloc[pltIndex,2] == 8], H1LatLong.iloc[pltIndex,1].loc[H1LatLong.iloc[pltIndex,2] == 8], "b*")
    plt.plot(H1LatLong.iloc[pltIndex,0].loc[H1LatLong.iloc[pltIndex,2] == 9], H1LatLong.iloc[pltIndex,1].loc[H1LatLong.iloc[pltIndex,2] == 9], "c*")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


    H1Salary = H1LatLong
    H1Salary['Salary'] = H1Info['PREVAILING_WAGE']
    H1Salary['FT'] = H1Info['FULL_TIME_POSITION']
    salaryMin = np.zeros(nClusters)
    salaryMax = np.zeros(nClusters)
    salaryMean = np.zeros(nClusters)
    salaryMedian = np.zeros(nClusters)
    salaryStd = np.zeros(nClusters)
    
    for k in range(0,nClusters):
        salaryClustered = H1Salary.loc[H1Salary['Cluster'] == k]
        salaryClustered = salaryClustered.loc[salaryClustered['FT'] == 'Y'] #We only consider full-time employment
        salaryClustered = salaryClustered.loc[salaryClustered['Salary'] <= 5000000] #We take out extreme outliers
        salaryClustered = salaryClustered.loc[salaryClustered['Salary'] > 0] # We take out non-paid positions
        salaryClustered = salaryClustered.dropna()
        salaryMin[k] = np.min(salaryClustered['Salary'].values)
        salaryMax[k] = np.max(salaryClustered['Salary'].values)
        salaryMean[k] = np.mean(salaryClustered['Salary'].values)
        salaryMedian[k] = np.median(salaryClustered['Salary'].values)
        salaryStd[k] = np.std(salaryClustered['Salary'].values)
    #Statistics
    salaryStats = pd.DataFrame()
    salaryStats['Mean'] = salaryMean.astype(int)
    salaryStats['Median'] = salaryMedian.astype(int)
    salaryStats['Std'] = salaryStd.astype(int)
    salaryStats['Min'] = salaryMin.astype(int)
    salaryStats['Max'] = salaryMax
    salaryStats
    

def learn():
    
    print("\t\t\tWHAT IS H1B VISA ???  \n\n")
    print("The H1B visa is an employment-based, non-immigrant visa for temporary workers. For this visa, an employer must offer a job in the US and apply for your H1B visa petition with the US Immigration Department. This approved petition is a work permit which allows you to obtain a visa stamp and work in the U.S. for that employer.\n\n")
    print("\t\t\tELIGIBILITY \n\n")
    print("The H1B visa is issued for a specialty occupation, requires theoretical and practical application of a body of specialized knowledge and requires the visa holder to have at least a Bachelors degree or its equivalent.")    
    print("\t\t\tKEY FACTS ABOUT H1B VISA\n\n")
    print("1. Since 2005, H-1B visas have been capped at 65,000 a year, plus an additional 20,000 visas for foreigners with a graduate degree from a U.S. academic institution")
    print("2. Demand for H-1B workers has boomed in recent years")
    print("3. More than half of all H-1B visas have been awarded to Indian nationals")
    print("4. H-1B visas are available for part-time as well as full-time positions.")


photo = tkinter.PhotoImage(file = "image.gif")
w = tkinter.Label(root, image=photo)
w.pack()

# tkinterinter button creation
button1=tkinter.Button(frame,text = "DATA SAMPLE DETAILS",command=datasample)
button1.pack()
button2=tkinter.Button(frame,text = "TOP 15 COMPANIES",command=topcompanies)
button2.pack()
button3=tkinter.Button(frame,text = "WAGE ANALYSIS",command=wageanalysis)
button3.pack()
button4=tkinter.Button(frame,text = "CITIES WITH MAXIMUM JOBS",command=maximumjobs)
button4.pack()
button5=tkinter.Button(frame,text = "PETITION ANALYSIS",command=petitionanalysis)
button5.pack()
button6=tkinter.Button(frame,text = "TOP JOB TITLES",command=topjobtitles)
button6.pack()
button9=tkinter.Button(frame,text = "EXTRA GRAPHS",command=extra)
button9.pack()
button10=tkinter.Button(frame,text = "CORELATION",command=corelation)
button10.pack()
button7 = tkinter.Button(frame, text = "LEARN ABOUT H1B VISA !!!",command=learn)
button7.pack()
button8 = tkinter.Button (frame, text = "EXIT THE TKINTER WINDOW !!!", command = root.destroy)
button8.pack()


root.mainloop()
