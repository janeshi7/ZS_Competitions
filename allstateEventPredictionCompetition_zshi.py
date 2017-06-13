'''
University of Chicago
Zhiyin Shi
Feb, 2017
'''

#Library & File Import
import csv
import itertools
import pandas as pd
import xgboost
from sklearn.feature_extraction import DictVectorizer

train_file = '/Users/JaneShi/Desktop/Allstate Competition/train.csv'
test_ids_file = '/Users/JaneShi/Desktop/Allstate Competition/test.csv'

# Read in data
with open(train_file) as file:
    reader = csv.DictReader(file, delimiter=',')
    raw_data = [row for row in reader]
print(raw_data[0])
print(len(raw_data))

with open(test_ids_file) as file:
    reader = csv.DictReader(file, delimiter=',')
    test_ids_data = [row for row in reader]
print(test_ids_data[0])
print(len(test_ids_data))

# Extract test data from raw train 
test_id_set = set([i['id'] for i in test_ids_data])
raw_test_data = [i for i in raw_data if i['id'] in test_id_set]

#Sort data
sorted_raw_test_data = sorted(raw_test_data, key = lambda x: (x['id'], int(x['timestamp'])))
sorted_raw_data = sorted(raw_data, key = lambda x: (x['id'], int(x['timestamp'])))

# 125 Feature construction function
def construct_features(instances, make_target = False):
    grouped_data = []
    for k, g in itertools.groupby(instances, key = lambda x : x['id']):
        group = list(g)
        
        if make_target == True:
            target_record = group.pop(-1)
        
        #Unique identifier (id) and Feature 1 (num_events)
        output_record = {'id': k, 'num_events': len(group)}
        
        #Feature 2 - 11: (Count frequency of event occurence)
        event_list = ['30018', '30021', '30024', '30027', '30039', '30042', '30045', '30048', '36003', '45003']
        event_freq = {}
        for i in event_list:
            event_freq['evt_freq_' + i] = 0   
        for i in group:
            event_freq['evt_freq_' + i['event']] += 1
        
        #Feature 12 - 21: (Position of last occurance of each event for each id)
        last_event_pos = {}
        for i in event_list:
            last_event_pos['last_event_pos_' + i] = 0
        for i in group:
            if int(i['timestamp']) > last_event_pos['last_event_pos_' + i['event']]:
                last_event_pos['last_event_pos_' + i['event']] = int(i['timestamp'])
                
        #Continue: Feature 12 - 21: Normalize the position 
        base = min(last_event_pos[i] for i in last_event_pos if last_event_pos[i] > 0)
        for i in last_event_pos:
            if last_event_pos[i] > 0:
                last_event_pos[i] = last_event_pos[i] - base + 1
        
        #Feature 22 - 121: (Frequency of pair-wise events occurance)
        pairwise_events = [p for p in itertools.product(event_list, repeat = 2)]
        pairwise_events_ct = {}
        for i in pairwise_events:
            pairwise_events_ct['pairwise_' + i[0] + '_' + i[1]] = 0
        prev = group[0]
        for i in group:
            if i['timestamp'] == prev['timestamp']:
                prev = i
                continue
            pairwise_events_ct['pairwise_' + prev['event'] + '_' + i['event']] += 1
            prev = i
        
        #Feature 122: (If this id has more than three events)
        if output_record['num_events'] < 4:
            output_record['if3event'] = 1
        else :
            output_record['if3event'] = 0
        
        #Feature 123: (first_event)           
        output_record['first_event'] = group[0]['event']

        #Feature 124: (last_event)
        output_record['last_event'] = group[-1]['event']

        #Feature 125: (middle_event)
        output_record['middle_event'] = group[int((len(group) - 1) // 2)]['event']

        #Collect all features in one signle dictionary
        output_record.update(event_freq)
        output_record.update(last_event_pos)
        output_record.update(pairwise_events_ct)

        #Make target
        if make_target == True:
            output_record['target_event'] = target_record['event']
        else:
            output_record['target_event'] = 'NA'

        grouped_data.append(output_record)
    return grouped_data

#Train and Test data feature construction
raw_data_trans = construct_features(sorted_raw_data, make_target = True)
test_trans = construct_features(sorted_raw_test_data, make_target = False)

#Convert test data to dataframe
test_trans_df = pd.DataFrame(test_trans)

#Seperate predictor and response variables
predictor_list = list(raw_data_trans[0].keys())
target_list = 'target_event'

predictor_list.remove('id')
predictor_list.remove('target_event')

train_predictors = [{key:value for key, value in i.items() if key in predictor_list} for i in raw_data_trans]
train_target = [i[target_list] for i in raw_data_trans]

test_predictors = [{key:value for key, value in i.items() if key in predictor_list} for i in test_trans]

# Convert categoricals to dummies (this results in 152 predictors)
dict_vec = DictVectorizer()
dict_vec.fit(train_predictors + test_predictors)

train_predictor_modeling = dict_vec.transform(train_predictors).toarray()
test_predictor_modeling = dict_vec.transform(test_predictors).toarray()

#Fit gradient boosting model with xgboost
model = xgboost.XGBClassifier(learning_rate = 0.03, max_depth = 4, n_estimators = 1000)
model.fit(train_predictor_modeling, train_target)

#Prediction on test data
test_prediction = model.predict_proba(test_predictor_modeling)

#Rename columns
classes = model.classes_.tolist()

for i in range(len(classes)):
    classes[i] = 'event_'+ str(classes[i])

test_prediction_df = pd.DataFrame(test_prediction)
test_prediction_df.index = test_trans_df.index
test_prediction_df.insert(0, 'id', test_trans_df['id'])
test_prediction_df.columns = ['id'] + classes

#Export as csv
test_prediction_df.to_csv('UC_ZS_test_pred_125feature_xgb.csv', header = True, index = False)
