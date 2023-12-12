import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


#Read in the partquet file from earlier and read in features worked on in Google Sheets
df = pd.read_parquet("daily.parquet")
feat_df = pd.read_csv('Listing_Features.csv')

#convert and filter as required
df_23 = df[df['year'] == 2023]
feat_df['id'] = pd.to_numeric(feat_df['id'], downcast = 'integer')

#Filter this down to only look at 2.5 -> 97.5% quantiles
min_price = df_23['price'].quantile([0.025]).iloc[0]
max_price = df_23['price'].quantile([0.975]).iloc[0]
to_use = df_23[(df_23['price'] >= min_price) & (df_23['price'] <= max_price)]

#Get the price for 2023 by listing and the number of days it was used
price_listing = df_23.groupby('listing_id').sum(numeric_only = True)['price']
no_days = df_23.groupby(['listing_id']).count()['days']

#Merge the targets with the features
to_engine = feat_df.merge(targets, left_on = 'id', right_index= True)

#Remove the fields that are not to be used here(This would be for targeting occupancy)
cols = to_engine.columns
cols = list(cols)
for i in ['longitude','latitude','price_x','availability_365','price_y']:
    cols.remove(i)
to_engine_occupancy = to_engine[cols]

#Remove the fields that are not to be used here(This would be for targeting 2023 costs)
cols = to_engine.columns
cols = list(cols)
for i in ['longitude','latitude','price_x','availability_365','days']:
    cols.remove(i)
to_engine_price = to_engine[cols]
###At this stage, I am able to push this to create the graphs in Tableau


#Setting up the handling of data here, so scaling and 
numeric_features = ["Star", "Bedrooms","Baths"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = ["neighbourhood_group", "room_type"]
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor)]
)

#From here the script will split into two diferent routes



"""
I will attempt to get a value for the link to occupancy in the below part of the script
"""

#Here we will remove some of the fields that were generated as they cause issues for the data
to_engine_occupancy = to_engine_occupancy[~to_engine_occupancy['Star'].isin(['No_Stars','New'])]
to_engine_occupancy['Bedrooms'] = to_engine_occupancy['Bedrooms'].replace('Studio',0)
to_engine_occupancy['Baths'] = to_engine_occupancy['Baths'].replace('Shared/Half',0.5)
to_engine_occupancy = to_engine_occupancy.drop(columns='id')
X = to_engine_occupancy[['Star','Bedrooms','Baths','neighbourhood_group','room_type']]
y = to_engine_occupancy[['days']]


X_out = clf.fit_transform(X)
s_scale = StandardScaler()
y_out = s_scale.fit_transform(y)


X_arr = X_out.toarray()
X_df = pd.DataFrame(X_arr)
tot = X_df.merge(pd.DataFrame(y_out), left_index=True, right_index=True)

corr_check = list(tot.corr().iloc[:,-1].abs())
corr_check.sort()
print(corr_check)

"""
You can see that no values here show a true correlation to the number of days booked out
[0.0037391344855829973,
 0.014644377313800487,
 0.01774663113991627,
 0.018479980331485936,
 0.02092499406112519,
 0.024169492592183418,
 0.02661521035644664,
 0.03328785946463829,
 0.03424626222149823,
 0.03946982794236165,
 0.03972414091168609,
 0.04029815368268775,
 0.04537675447588037,
 0.054746799868620126,
 0.06253632676313922,
 0.06387568643525353,
 0.07120168315131964,
 0.07773008387727512,
 0.10286976579148703,
 1.0]
 """
 


"""
I will attempt to get a value for the link to earnings for 2023 in the below part of the script
"""

 
to_engine_price = to_engine_price[~to_engine_price['Star'].isin(['No_Stars','New'])]
to_engine_price['Bedrooms'] = to_engine_price['Bedrooms'].replace('Studio',0)
to_engine_price['Baths'] = to_engine_price['Baths'].replace('Shared/Half',0.5)
to_engine_price = to_engine_price.drop(columns='id')
X = to_engine_price[['Star','Bedrooms','Baths','neighbourhood_group','room_type']]
y = to_engine_price[['price_y']]

s_imp = StandardScaler()
y_out = s_imp.fit_transform(y)

X_out = clf.fit_transform(X)
X_arr = X_out.toarray()
X_df = pd.DataFrame(X_arr)
tot = X_df.merge(pd.DataFrame(y_out), left_index=True, right_index=True)

corr_check = list(tot.corr().iloc[:,-1].abs())
corr_check.sort()
corr_check
 
 
 """
 To Price there is a greater opportunity, but still nothing of note
 [0.005413448873969543,
 0.015427846797848836,
 0.01920734549796296,
 0.02206910734699069,
 0.04103644952885253,
 0.04272621974282977,
 0.055156070621807995,
 0.05695152052107367,
 0.057891979087885334,
 0.059242631010585044,
 0.06367776149485459,
 0.06458694983425836,
 0.07626978190153454,
 0.08506554173276354,
 0.10078817390516588,
 0.15290416755648362,
 0.22157526381902098,
 0.2477254592088307,
 0.2561108489845762,
 1.0]
 """
