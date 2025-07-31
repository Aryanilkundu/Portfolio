negative_keywords = [
    # Wickets
    "out", "bowled", "caught", "lbw", "stumped", "hit wicket", "run out",
    "edge taken", "top edge", "chopped on", "nicked behind",
    
    # Shot Mistakes
    "poor shot", "rash shot", "unnecessary shot", "mistimed", "miscued",
    "airborne and gone", "bad decision",
    
    # Pressure Situations
    "pressure builds", "dot balls", "struggling", "slow scoring", "tight over",
    "dry spell", "collapse",
    
    # Bowling Praise (opponent)
    "brilliant yorker", "sharp turn", "beaten", "unplayable", "deadly spell",
    "nagging line", "great over",
    
    # Injuries or Issues
    "injury", "limping", "cramp", "needs treatment", "retired hurt"
]
positive_keywords = [
    # Runs Scored
    "four", "boundary", "six", "maximum", "runs", "single", "double", "triple",
    "driven", "swept", "cut", "flicked", "pulled", "lofted", "hit hard",
    
    # Shot Quality
    "beautifully timed", "cracking shot", "glorious", "perfect placement",
    "pierces the gap", "sweet timing",
    
    # Milestones
    "fifty", "half-century", "hundred", "century", "maiden hundred", "milestone",
    
    # Momentum Words
    "accelerating", "good partnership", "building innings", "taking charge",
    "in control", "positive intent",
    
    # Bowling Errors
    "no ball", "wide", "free hit", "overthrows", "misfield", "dropped catch",
    
    # Extras Benefiting Batting
    "leg byes", "byes", "overstep", "bonus run",
    
    # Commentator Praise
    "excellent shot", "top-class", "magnificent", "sensational", "standout"
]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')
# import tensorflow as tf
df = pd.read_csv(r"C:\Users\ARYANIL TANISHA\Downloads\1031665_COMMENTARY.csv")
data = df[df['Innings'] == "2nd innings"]
data['Over_Ball_new'] =( data['Over_number']-1).astype(str) + '.' + data['Over_ball'].astype(str)

# print(data.head())
sentimentdata = data[["Over_Ball_new","Commentary"]]
sentimentdata["ID"] = sentimentdata.index 
stop = stopwords.words('english')
sentimentdata["Commentary"].apply(lambda txt: ' '.join([word for word in str(txt).split() if word not in (stop)]))
sia = SentimentIntensityAnalyzer()

# positive_keywords = ['four', 'six', 'boundary', 'no ball', 'free hit']
# negative_keywords = ['out', 'caught', 'bowled', 'lbw', 'dot ball']

def cricket_sentiment_adjusted(commentary):
    commentary = commentary.lower()
    score = sia.polarity_scores(commentary)["compound"]

    for word in positive_keywords:
        if word in commentary:
            return abs(score)  # benefit to batting team
    for word in negative_keywords:
        if word in commentary:
            return -abs(score)  # harm to batting team

    return 0.1*score  # neutral if unclear

result = {}

for i, row in tqdm(sentimentdata.iterrows(), total=len(sentimentdata)):
    text = str(sentimentdata.loc[i, "Commentary"])
    matchno = sentimentdata.loc[i, "ID"]
    result[matchno] = cricket_sentiment_adjusted(text)


sentimentdata["adjusted_sentiment"] = pd.Series(result)
data['adjusted_sentiment'] = sentimentdata['adjusted_sentiment'].values



# Constants
data_1st = df[df['Innings'] == "1st innings"]
target = data_1st["Innings_runs"].iloc[-1]
total_balls = 120

data["Balls_left"] = total_balls - data["Innings_balls"]
data["Runs_left"] = target - data["Innings_runs"]

# Initial Required Run Rate
initial_rr = target / total_balls

# Current Required Run Rate
data["Req_Rate"] = (data["Runs_left"] / data["Balls_left"]) * 6
data["Req_Rate"] = data["Req_Rate"].replace([np.inf, -np.inf], np.nan).fillna(0)

# PI calculation
data["PI"] = (data["Req_Rate"] / initial_rr) * 100 + \
           (data["Balls_left"] / total_balls) * \
           (data["Runs_left"] / target)

scaler = MinMaxScaler()
data["PI_scaled"] = scaler.fit_transform(data["PI"].values.reshape(-1, 1))
# data["rolling_momentum"] = data["adjusted_sentiment"].ewm(span=6, adjust=False).mean()
data["rolling_momentum"] = data["adjusted_sentiment"].rolling(window=12, min_periods=1).mean()

plt.figure(figsize=(12, 6))

# Use cumulative ball number on the x-axis
x = data["Innings_balls"]  # assumes this counts 1..120 for the innings

plt.plot(x, data['PI_scaled'], label='Pressure Index')
plt.plot(x, data['rolling_momentum'], label='Momentum')

# Tick positions every over (every 6 balls) and labels as over numbers
last_ball = int(x.max())
tick_pos = np.arange(6, last_ball + 1, 6)
tick_labels = (tick_pos // 6).astype(int)

plt.xticks(tick_pos, tick_labels)
plt.xlabel("Over")
plt.ylabel("Momentum / Scaled PI")
plt.title("Momentum Comparison")
plt.legend()
plt.tight_layout()
plt.show()
