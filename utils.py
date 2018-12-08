
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import pandas as pd
import seaborn as sns
# sns.set(style="darkgrid", font_scale=2)


def load_clean_data():
	# Source: https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results
	df_raw = pd.read_csv('athlete_events.csv')
	# df_noc = pd.read_csv('noc_regions.csv')    # Region abbreviations

	# Only consider data after 1950
	df = df_raw.loc[df_raw.Year > 1950]

	return remove_nans(df)

def remove_nans(data):
    """ Remove rows with NaNs EXCEPT medal column """
    assert isinstance(data, pd.DataFrame)

    cols = ['Weight', 'Age', 'Sex', 'Height']
    for c in cols:
        if c != 'Medal':
            data = data.drop(data[data[c].isnull()].index, axis=0)
   
    return data

def plot_sum_win_games(df):
	""" Plots the summer vs winter games plot """
	games = np.unique(df.Games)
	num_athletes = [np.unique(df[df.Games==g].ID).size for g in games]
	color = ['r' if i.split()[1]=='Summer' else 'b' for i in games]
	df_sum_win = pd.DataFrame([games, num_athletes, color], index=['Year', 'Athletes', 'Color']).T
	fig, ax = plt.subplots(figsize=(20,8))
	sum_win_plot = sns.barplot(x='Year', y='Athletes', data=df_sum_win, ax=ax, palette=color)
	sum_win_plot.set_xticklabels(labels=[g.split()[0] for g in games], rotation=75)
	ax.set_title('Athletes in Summer vs Winter Games')
	ax.legend(handles=[pat.Patch(color='r', label='Summer Games'), 
	                  pat.Patch(color='b', label='Winter Games')])
	plt.show()

def plot_age(df):
	""" Plot age vs medals over time """
	g_all = df.loc[df["Medal"] == "Gold"]
	s_all = df.loc[df["Medal"] == "Silver"]
	b_all = df.loc[df["Medal"] == "Bronze"]
	top_3 = df.loc[[i == "Gold" or i == "Silver" or i == "Bronze" for i in df.Medal]]

	age_year = sns.relplot(x = "Year", y = "Age", hue = "Medal", kind = "line", palette = ["#cd7f32", "#ffcf40", "#c0c0c0"], ci = 50, height = 8, aspect = 1.8, data = top_3)
	plt.title("Mean Age Trends of Medalists")
	plt.show()

def plot_male_age_hw(df):
	""" Plot age vs height/weight plots for male """
	# Sort by highest medals per athlete
	events = np.unique(df.Sport)
	num_medals = [df.loc[df.Sport==e].Medal.notnull().sum() /
	              df.loc[df.Sport==e].Medal.size for e in events]  # find number of medals in each event
	event_medals = dict(zip(events, num_medals))  # combine
	sorted_event_medals = [[e,event_medals[e]] for e in sorted(event_medals, key=event_medals.get, reverse=True)]  # sort by medals

	da = df.loc[df['Medal'].isin(('Bronze', 'Silver', 'Gold'))]
	select = [u[0] for u in sorted_event_medals[:30]]
	dw = da.loc[da['Sport'].isin(set(select))]
	dss = dw.groupby(['Sport', 'Sex'])
	hight = dss['Height'].mean()
	weight = dss['Weight'].mean()
	age = dss['Age'].mean()
	ready = []
	# Analyze Men
	for sp in select:
	    try:
	        temp = [sp, hight[sp, 'M'] / weight[sp, 'M'], age[sp, 'M']]
	        ready.append(temp)
	    except KeyError:
	        select.remove(sp)
	df_male = pd.DataFrame(ready, columns=['Sport', 'Height/Weight', 'Age'])
	g = sns.FacetGrid(df_male, hue='Sport', height=8, aspect=1.5)
	g.map(plt.scatter, 'Height/Weight', 'Age', s=200)
	g.add_legend()
	g.ax.set_title('(Male) Age vs Height/Weight')
	plt.show()

def plot_female_age_hw(df):
	""" Plot age vs height/weight plots for females """
	# Sort by highest medals per athlete
	events = np.unique(df.Sport)
	num_medals = [df.loc[df.Sport==e].Medal.notnull().sum() /
	              df.loc[df.Sport==e].Medal.size for e in events]  # find number of medals in each event
	event_medals = dict(zip(events, num_medals))  # combine
	sorted_event_medals = [[e,event_medals[e]] for e in sorted(event_medals, key=event_medals.get, reverse=True)]  # sort by medals

	da = df.loc[df['Medal'].isin(('Bronze', 'Silver', 'Gold'))]
	select = [u[0] for u in sorted_event_medals[:30]]
	dw = da.loc[da['Sport'].isin(set(select))]
	dss = dw.groupby(['Sport', 'Sex'])
	hight = dss['Height'].mean()
	weight = dss['Weight'].mean()
	age = dss['Age'].mean()
	ready = []
	# Analyze Women
	for sp in select:
	    try:
	        temp = [sp, hight[sp, 'F'] / weight[sp, 'F'], age[sp, 'F']]
	        ready.append(temp)
	    except KeyError:
	        select.remove(sp)
	df_female = pd.DataFrame(ready, columns=['Sport', 'Height/Weight', 'Age'])
	g = sns.FacetGrid(df_female, hue='Sport', height=10, aspect=2)
	g.map(plt.scatter, 'Height/Weight', 'Age', s=200)
	g.add_legend()
	g.ax.set_title('(Female) Age vs Height/Weight')
	plt.show()

def plot_gender(df):
	""" Plot trend of gender over time """
	ds = df.groupby(['Year', 'Sex'])
	y = set(df['Year'].values)
	content = []
	for yy in y:
	    content.append(ds.size().loc[yy, 'F'] / ds.size().loc[yy, 'M'])
	plt.figure(figsize=(15,8))
	pp = sns.pointplot(list(y), content, alpha=0.8)
	pp.set_xticklabels(labels=y, rotation=75)
	plt.xlabel('Year')
	plt.ylabel('Ratio of Female to Male')
	plt.title('Ratio of Female to Male overtime')
	plt.show()

def plot_pie_chart(df):
	""" Plot pie chart for medals """
	top_3 = df.loc[[i == "Gold" or i == "Silver" or i == "Bronze" for i in df.Medal]]
	labels = 'Top 10', 'Other'
	sizes = [top_3['NOC'].value_counts().iloc[:10].sum()/top_3['NOC'].value_counts().sum()*100, 100-top_3['NOC'].value_counts().iloc[:10].sum()/top_3['NOC'].value_counts().sum()*100]
	explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

	fig1, ax1 = plt.subplots()
	ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
	        shadow=True, startangle=90)
	ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

	plt.show()

def plot_heatmap(df):
	top_3 = df.loc[[i == "Gold" or i == "Silver" or i == "Bronze" for i in df.Medal]]
	sports = list(set(top_3.Sport))
	countries = {"USA", "URS", "GER", "AUS", "RUS", "ITA", "GDR", "CHN", "CAN", "GBR"}
	nation_sport = pd.DataFrame(index = countries, columns = sports, data = np.zeros((len(countries), len(sports))))

	for i in sports:
	    for j in countries:
	        try:
	            nation_sport.loc[j,i] = (top_3.loc[top_3.NOC == j].loc[top_3.Sport == i].shape[0]) / (top_3.loc[top_3.Sport == i].shape[0])
	        except:
	            nation_sport.loc[j,i] = 0
	plt.figure(figsize=(30,12))
	sns.heatmap(nation_sport, cmap = "GnBu")
	plt.title("Nations' Medal Winning Percentage for Each Sport")
	plt.show()

def plot_USA(df):
	top_3 = df.loc[[i == "Gold" or i == "Silver" or i == "Bronze" for i in df.Medal]]
	sports = list(set(top_3.Sport))
	USA_winning_rate = pd.DataFrame(index = range(len(sports)), columns = ["Sport", "Overall", "Gold Medal"], data = np.zeros((len(sports), 3)))
	for i in range(len(sports)):
	    USA_winning_rate.iloc[i,0] = sports[i]

	for i in range(len(sports)):
	    try:
	        USA_winning_rate.iloc[i, 1] = (top_3.loc[top_3.NOC == "USA"].loc[top_3.Sport == sports[i]].shape[0]) / (top_3.loc[top_3.Sport == sports[i]].shape[0])
	    except:
	        USA_winning_rate.iloc[i, 1] = 0
	    
	    try:
	        USA_winning_rate.iloc[i, 2] = (top_3.loc[top_3.NOC == "USA"].loc[top_3.Sport == sports[i]].loc[top_3.Medal == "Gold"].shape[0]) / (top_3.loc[top_3.Sport == sports[i]].shape[0])
	    except:
	        USA_winning_rate.iloc[i, 2] = 0

	plt.figure(figsize=(25,12))
	sns.set_color_codes("pastel")
	sns.barplot(x="Overall", y = "Sport", data=USA_winning_rate.sort_values("Overall", ascending=False).iloc[:20, :], label = "Silver and Bronze", color="b")

	sns.set_color_codes("muted")
	sns.barplot(x="Gold Medal", y="Sport", data=USA_winning_rate.sort_values("Overall", ascending=False).iloc[:20, :], label = "Gold Medals", color="#ffcf40")
	plt.title("USA Medal and Gold Medal Winning Percentage Top 20")
	plt.xlabel("Winning Percentage")
	plt.legend()
	plt.show()

def plot_CHN(df):
	top_3 = df.loc[[i == "Gold" or i == "Silver" or i == "Bronze" for i in df.Medal]]
	sports = list(set(top_3.Sport))
	CHN_winning_rate = pd.DataFrame(index = range(len(sports)), columns = ["Sport", "Overall", "Gold Medal"], data = np.zeros((len(sports), 3)))
	for i in range(len(sports)):
	    CHN_winning_rate.iloc[i,0] = sports[i]

	for i in range(len(sports)):
	    try:
	        CHN_winning_rate.iloc[i, 1] = (top_3.loc[top_3.NOC == "CHN"].loc[top_3.Sport == sports[i]].shape[0]) / (top_3.loc[top_3.Sport == sports[i]].shape[0])
	    except:
	        CHN_winning_rate.iloc[i, 1] = 0
	    
	    try:
	        CHN_winning_rate.iloc[i, 2] = (top_3.loc[top_3.NOC == "CHN"].loc[top_3.Sport == sports[i]].loc[top_3.Medal == "Gold"].shape[0]) / (top_3.loc[top_3.Sport == sports[i]].shape[0])
	    except:
	        CHN_winning_rate.iloc[i, 2] = 0

	plt.figure(figsize=(25,12))
	sns.set_color_codes("pastel")
	sns.barplot(x="Overall", y = "Sport", data=CHN_winning_rate.sort_values("Overall", ascending=False).iloc[:20, :], label = "Silver and Bronze", color="b")

	sns.set_color_codes("muted")
	sns.barplot(x="Gold Medal", y="Sport", data=CHN_winning_rate.sort_values("Overall", ascending=False).iloc[:20, :], label = "Gold Medals", color="#ffcf40")
	plt.title("China Medal and Gold Medal Winning Percentage Top 20")
	plt.xlabel("Winning Percentage")
	plt.legend()
	plt.show()

def plot_CAN(df):
	top_3 = df.loc[[i == "Gold" or i == "Silver" or i == "Bronze" for i in df.Medal]]
	sports = list(set(top_3.Sport))
	CAN_winning_rate = pd.DataFrame(index = range(len(sports)), columns = ["Sport", "Overall", "Gold Medal"], data = np.zeros((len(sports), 3)))
	for i in range(len(sports)):
	    CAN_winning_rate.iloc[i,0] = sports[i]

	for i in range(len(sports)):
	    try:
	        CAN_winning_rate.iloc[i, 1] = (top_3.loc[top_3.NOC == "CAN"].loc[top_3.Sport == sports[i]].shape[0]) / (top_3.loc[top_3.Sport == sports[i]].shape[0])
	    except:
	        CAN_winning_rate.iloc[i, 1] = 0
	    
	    try:
	        CAN_winning_rate.iloc[i, 2] = (top_3.loc[top_3.NOC == "CAN"].loc[top_3.Sport == sports[i]].loc[top_3.Medal == "Gold"].shape[0]) / (top_3.loc[top_3.Sport == sports[i]].shape[0])
	    except:
	        CAN_winning_rate.iloc[i, 2] = 0

	plt.figure(figsize=(25,12))
	sns.set_color_codes("pastel")
	sns.barplot(x="Overall", y = "Sport", data=CAN_winning_rate.sort_values("Overall", ascending=False).iloc[:20, :], label = "Silver and Bronze", color="b")

	sns.set_color_codes("muted")
	sns.barplot(x="Gold Medal", y="Sport", data=CAN_winning_rate.sort_values("Overall", ascending=False).iloc[:20, :], label = "Gold Medals", color="#ffcf40")
	plt.title("Canada Medal and Gold Medal Winning Percentage Top 20")
	plt.xlabel("Winning Percentage")
	plt.legend()
	plt.show()

def get_male_df(df):
	# Sort by highest medals per athlete
	events = np.unique(df.Sport)
	num_medals = [df.loc[df.Sport==e].Medal.notnull().sum() /
	              df.loc[df.Sport==e].Medal.size for e in events]  # find number of medals in each event
	event_medals = dict(zip(events, num_medals))  # combine
	sorted_event_medals = [[e,event_medals[e]] for e in sorted(event_medals, key=event_medals.get, reverse=True)]  # sort by medals
	da = df.loc[df['Medal'].isin(('Bronze', 'Silver', 'Gold'))]
	select = [u[0] for u in sorted_event_medals[:30]]
	dw = da.loc[da['Sport'].isin(set(select))]
	dss = dw.groupby(['Sport', 'Sex'])
	hight = dss['Height'].mean()
	weight = dss['Weight'].mean()
	age = dss['Age'].mean()
	ready = []
	# Analyze Men
	for sp in select:
	    try:
	        temp = [sp, hight[sp, 'M'] / weight[sp, 'M'], age[sp, 'M']]
	        ready.append(temp)
	    except KeyError:
	        select.remove(sp)
	df_male = pd.DataFrame(ready, columns=['Sport', 'Height/Weight', 'Age'])
	return df_male

def get_female_df(df):
	# Sort by highest medals per athlete
	events = np.unique(df.Sport)
	num_medals = [df.loc[df.Sport==e].Medal.notnull().sum() /
	              df.loc[df.Sport==e].Medal.size for e in events]  # find number of medals in each event
	event_medals = dict(zip(events, num_medals))  # combine
	sorted_event_medals = [[e,event_medals[e]] for e in sorted(event_medals, key=event_medals.get, reverse=True)]  # sort by medals
	da = df.loc[df['Medal'].isin(('Bronze', 'Silver', 'Gold'))]
	select = [u[0] for u in sorted_event_medals[:30]]
	dw = da.loc[da['Sport'].isin(set(select))]
	dss = dw.groupby(['Sport', 'Sex'])
	hight = dss['Height'].mean()
	weight = dss['Weight'].mean()
	age = dss['Age'].mean()
	ready = []
	# Analyze Women
	ready = []
	for sp in select:
	    try:
	        temp = [sp, hight[sp, 'F'] / weight[sp, 'F'], age[sp, 'F']]
	        ready.append(temp)
	    except KeyError:
	        select.remove(sp)
	df_female = pd.DataFrame(ready, columns=['Sport', 'Height/Weight', 'Age'])
	return df_female


def find_sport(gender, age, height, weight, df):
    """ Finds the closest sport based on gender, age, height, and weight """
    assert gender in ['M', 'F'], 'Gender can only be "M" or "F"'
    
    query = np.array([age, height/weight])
    if gender == 'M':
        df_male = get_male_df(df)
        sport = [np.linalg.norm(query - np.array([df_male.iloc[i].Age, 
                                                  df_male.iloc[i]['Height/Weight']])) for i in range(df_male.shape[0])]
        result = df_male.iloc[np.argmin(sport)].Sport
    else:
        df_female = get_female_df(df)
        sport = [np.linalg.norm(query - np.array([df_female.iloc[i].Age, 
                                                  df_female.iloc[i]['Height/Weight']])) for i in range(df_female.shape[0])]
        result = df_female.iloc[np.argmin(sport)].Sport
    
    return result
