import pandas as pd

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


