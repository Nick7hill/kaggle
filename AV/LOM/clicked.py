
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



train_df=pd.read_csv('train.csv')
campaign=pd.read_csv('campaign_data.csv')
test_df=pd.read_csv('test_BDIfz5B.csv')
sub=pd.read_csv('sample_submission_4fcZwvQ.csv')

train_merge=pd.merge(train_df,campaign,on='campaign_id',how='left')
test_merge=pd.merge(test_df,campaign,on='campaign_id',how='left')

least_freq_days_in_clicked=[0,1,2,3,4,5]

len_train=len(train_merge)

clicked=train_merge[train_merge.is_click==1]
top_clickers=clicked['user_id'].value_counts()[:100].index


y=train_merge['is_click']

train_merge.drop(['is_open','is_click'],axis=1,inplace=True)

train_merge=train_merge.append(test_merge)
print("train and test appended..")


def prep_data( df ):
    le=LabelEncoder()
    print("preparing the dataset ..")
    df['hour'] = pd.to_datetime(df.send_date).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.send_date).dt.day.astype('uint8')
    df['internals']=df['total_links']+df['no_of_images']+df['no_of_sections']
    df['received'] = df.groupby('user_id')['user_id'].transform('count')
    
    df['count_day']=df.groupby('day')['day'].transform('count')
    df['count_day'] = np.where(df.day.isin([0,1,2,3,4,5]),0, df['count_day'])
    
    df['sub_count']=df.groupby('subject')['subject'].transform('count')
    df['body_count']=df.groupby('email_body')['email_body'].transform('count')
    
    df['top_clickers']=0
    df['top_clickers']=np.where(df.user_id.isin(top_clickers),1,df['top_clickers'])
    
    
    
    df['com_count']=df.groupby('communication_type')['communication_type'].transform('count')
    df.drop(['send_date','id','no_of_internal_links','no_of_sections','email_url','no_of_images','subject','email_body',
           'no_of_sections','total_links'],axis=1,inplace=True)       
   
    df['communication_id']=le.fit_transform(df['communication_type'])
    df['campaign']=le.fit_transform(df['campaign_id'])
    df.drop(['campaign_id','communication_type'],axis=1,inplace=True)
    print("Finished preparing the data..")
    return( df )

print("preparing ........")
train_merge=prep_data(train_merge)
print ("Done...")

print("separating the train and test set..")
train_df=train_merge[:len_train]
test_df= train_merge[len_train:]

print("preparing training and validation set")
X_train,X_val,y_train,y_val=train_test_split(train_df,y,test_size=0.05, shuffle=False)
print("train and validation set prepared")


#defining the model features
import lightgbm as lgb
metrics = 'auc'
lgb_params = {
        'boosting_type': 'rf',
        'objective': 'binary',
        'metric':metrics,
        'learning_rate': 0.001,
        'num_leaves': 6,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'nthread': 8,
        'verbose': 0,
        'scale_pos_weight':98.75, # because training data is extremely unbalanced 
        'metric':metrics
}
predictors = ['user_id','hour','campaign','internals','received','top_clickers','com_count','count_day','sub_count','body_count']
categorical = ['hour','campaign','user_id','received','com_count','top_clickers','count_day','sub_count','body_count']




MAX_ROUNDS = 1000
EARLY_STOP = 50
OPT_ROUNDS = 680


num_boost_round=MAX_ROUNDS
early_stopping_rounds=EARLY_STOP

xgtrain = lgb.Dataset(X_train[predictors].values, label=y_train.values,
                              feature_name=predictors,
                              categorical_feature=categorical
                              )


xgvalid = lgb.Dataset(X_val[predictors].values, label=y_val.values,
                              feature_name=predictors,
                              categorical_feature=categorical
                              )

print("Model Started Training:")
evals_results = {}

bst = lgb.train(lgb_params,xgtrain, 
                         valid_sets= [xgvalid], 
                         valid_names=['valid'], 
                         evals_result=evals_results, 
                         num_boost_round=num_boost_round,
                         early_stopping_rounds=early_stopping_rounds,
                         verbose_eval=10, 
                         feval=None)

xgtrain = lgb.Dataset(train_df[predictors].values, label=y.values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
  

bst = lgb.train(lgb_params, xgtrain, num_boost_round=num_boost_round,verbose_eval=10,feval=None)
                     
n_estimators = bst.best_iteration
print("\nModel Report")
print("n_estimators : ", n_estimators)
print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

print("Predicting...")
sub['is_click'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv('outfileFull.csv', index=False, float_format='%.9f')
print("done...")
print(sub.info())"""
