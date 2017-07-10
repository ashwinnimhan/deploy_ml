from flask import Flask, request, jsonify
from preprocess import *

def process_rec(r):
	tips_df = r[(r['Payment_type'] == 1) & (r['Total_amount'] > 0) & (r['Tip_amount'] > 0)]
	tips_df = tips_df[(tips_df['Trip_distance'] > 0) & (tips_df['Trip_distance'] < 100)]

	tips_df['Fare_amount'] = tips_df['Total_amount'] - tips_df['Tip_amount']
	tips_df  = tips_df[tips_df['Fare_amount'] > 0]

	cols = ['Pickup_pt', 'Dropoff_pt', 'Pickup_hr', 'Dropoff_hr', 'Pickup_day', 'Dropoff_day', 'Trip_time']
	tips_df[cols] = tips_df.apply(lambda row: (process(row)), axis=1)

	cols = ['D_Time_bin', 'D_Time_num', 'D_Time_cos']
	tips_df[cols] = tips_df.apply(lambda row: (time_features(row['Lpep_dropoff_datetime'])), axis=1)

	tips_df['VendorID'] = tips_df['VendorID'].map({1:0, 2:1})
	tips_df['Dropoff_is_weekday'] = (tips_df['Dropoff_day']/5).astype('int')
	tips_df['Average_speed'] = tips_df['Trip_distance'] / tips_df['Trip_time']
	tips_df['Average_speed'] = tips_df['Average_speed'].replace([np.inf, -np.inf], 0)
	tips_df['Average_speed'] = tips_df['Average_speed'].fillna(0)

	tips_df.drop(['lpep_pickup_datetime',
	              'Lpep_dropoff_datetime',
	              'Payment_type',
	              'Total_amount',
	              'Tip_amount',
	              'Ehail_fee',
	              'Store_and_fwd_flag'], 
	             inplace=True, axis=1)

	return tips_df

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
	req_data = request.get_json()
	df = pd.DataFrame(req_data, index=[0])
	query = process_rec(df)

	if query.shape[0] != 0:
		prediction = clf.predict(query)
	else:
		prediction = 0

	return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
	with open('gbr-model.pkl', 'rb') as f:
		clf = pickle.load(f)
		print('Model details: ', clf)
	
	app.run(host='0.0.0.0', port=8080)
