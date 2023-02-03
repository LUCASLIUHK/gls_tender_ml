import pandas as pd
import numpy as np
import SQL_connect
from xgboost import XGBRegressor, DMatrix, cv, train, plot_tree, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, mean_absolute_percentage_error as mape, r2_score
import matplotlib.pylab as plt
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
import glspred
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval


def tune_param_rs(x_train, y_train, x_test, y_test, test_size, params):
    scoring = ['neg_mean_absolute_percentage_error']
    kfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    random_search = RandomizedSearchCV(estimator=xgb,
                                       param_distributions=params,
                                       n_iter=50,
                                       scoring=scoring,
                                       refit=scoring[0],
                                       n_jobs=-1,
                                       cv=kfold,
                                       verbose=1)

    # fit data
    random_search_res = random_search.fit(x_train, y_train)
    # Print the best score and the corresponding hyperparameters
    print(f'MAPE: {-random_search_res.best_score_:.4f}')
    print(f'Hyperparameters: {random_search_res.best_params_}')

    # apply tuned params
    param_tuned = random_search_res.best_params_
    xgb_tuned = train(params=param_tuned, dtrain=DMatrix(x_train, label=y_train), num_boost_round=500)
    pred_train, pred_test = xgb_tuned.predict(train_data), \
                            xgb_tuned.predict(test_data)
    mape_train, mape_test = mape(y_train, pred_train), \
                            mape(y_test, pred_test)
    res_tuned = [mape_train, mape_test] + [param_tuned[param] for param in key_params]
    res_df.loc[len(res_df)] = res_tuned

    return random_search_res, mape_train, mape_test


dbconn = SQL_connect.DBConnectionRS()

# read in data
gls = dbconn.read_data('''  select *
                            from data_science.sg_land_bidding_filled_features_with_comparable_price
                                left join data_science.sg_land_bidding_psm_price_hedonic_index_2022
                                using (year_launch)
                                left join data_science.sg_land_bidding_total_price_hedonic_index_2022
                                using (year_launch)
                            ;''')

# read in data for prediction
# pred = dbconn.read_data(''' select *
#                             from data_science.sg_gls_land_parcel_for_prediction
#                             ;''')

# read in infra table
infra = dbconn.read_data(''' select * from data_science.sg_land_parcel_distance_to_infrastructure''')

# read in nearby parcels data
dist_parcels = dbconn.read_data(''' select land_parcel_id_a as land_parcel_id, min(distance_m) as dist_to_nearest_parcel_launched_past_6m
                                    from data_science.sg_gls_pairwise_nearby_land_parcels
                                    where launch_time_diff_days >= 0
                                    and launch_time_diff_days <= 180
                                    and land_parcel_id_a != land_parcel_id_b 
                                    group by 1
                                    ;''')

nearby_parcels = dbconn.read_data('''   select land_parcel_id_a as land_parcel_id, count(*) as num_nearby_parcels_3km_past_6m
                                        from data_science.sg_gls_pairwise_nearby_land_parcels
                                        where launch_time_diff_days >= 0
                                        and launch_time_diff_days <= 180
                                        and land_parcel_id_a != land_parcel_id_b 
                                        and distance_m <= 3000
                                        group by 1
                                        ;''')

dist_proj = dbconn.read_data('''select land_parcel_id , min(distance) as dist_to_nearest_proj
                                from data_science.sg_land_parcel_filled_info_distance_to_project
                                where land_use_big = proj_devt_class
                                and land_launch_year >= proj_completion_year
                                and distance >= 10
                                group by 1
                                ;''')

nearby_proj = dbconn.read_data('''  select land_parcel_id , count(project_dwid) as num_proj_nearby_2km_5yr
                                    from data_science.sg_land_parcel_filled_info_distance_to_project
                                    where land_use_big = proj_devt_class
                                    and distance > 10
                                    and distance < 2000
                                    and land_launch_year - proj_completion_year > 0
                                    and land_launch_year - proj_completion_year < 5
                                    group by 1;''')

# full data
gls = gls.merge(dist_parcels, how='left', on='land_parcel_id')\
    .merge(nearby_parcels, how='left', on='land_parcel_id')\
    .merge(infra, how='left', on='land_parcel_id')\
    .merge(dist_proj, how='left', on='land_parcel_id')\
    .merge(nearby_proj, how='left', on='land_parcel_id')

# predicting data
# pred = pred.merge(dist_parcels, how='left', on='land_parcel_id')\
#     .merge(nearby_parcels, how='left', on='land_parcel_id')\
#     .merge(infra, how='left', on='land_parcel_id')\
#     .merge(dist_proj, how='left', on='land_parcel_id')\
#     .merge(nearby_proj, how='left', on='land_parcel_id')
pred = gls[gls.sg_gls_id.astype(str).str[-10:] == '0'*10]

# training data
gls = gls.drop(pred.index, axis=0)

# select target
gls['price_psm_real'] = gls.successful_price_psm_gfa / gls.hi_price_psm_gfa
target = 'price_psm_real'

# select features
gls['num_winners'] = gls.tenderer_name_1st.apply(lambda x: len(x.split('|')))
gls['joint_venture'] = gls.num_winners.apply(lambda x: 1 if x > 1 else 0)

cat_cols = ['region',
            'zone',
            'devt_class',
            'devt_type',
            'source'
            ]
num_cols = ['latitude',
            'longitude',
            'site_area_sqm',
            'lease_term',
            'gpr',
            'num_bidders',
            'joint_venture',
            'year_launch',
            'timediff_launch_to_close',
            'proj_num_of_units',  # should be dynamic
            'proj_max_floor',  # should be dynamic
            'num_nearby_parcels_3km_past_6m',
            # 'num_of_nearby_completed_proj_200m',  # calculate manually using coordinates
            'num_proj_nearby_2km_5yr',
            'num_mrt_1km',
            'num_bus_stop_500m',
            'num_school_1km',
            'dist_to_nearest_parcel_launched_past_6m',
            'dist_to_nearest_proj',
            'dist_to_cbd',
            'dist_to_mrt',
            'dist_to_bus_stop',
            'dist_to_school',
            'comparable_price_psm_gfa'
            ]
feature_cols = cat_cols + num_cols
cols = feature_cols + [target]

# pre-process
gls = gls.sort_values(by=['year_launch', 'month_launch', 'date_launch'])
gls = gls.dropna(subset=[target]).reset_index(drop=True)
# fillna for certain cols
gls = gls.fillna(pd.DataFrame(np.zeros((gls.shape[0], 3)),
                              columns=['num_mrt_1km',
                                       'num_bus_stop_500m',
                                       'num_school_1km']
                              )
                 )
gls = gls.fillna(pd.DataFrame(np.zeros((gls.shape[0], 1)),
                              columns=['num_nearby_parcels_3km_past_6m'])
                 )

# split data by randomly selecting most recent records as testing data
time_threshold = 2012
gls_dummy = pd.get_dummies(gls[cols])
gls_train = gls_dummy[gls_dummy.year_launch < time_threshold]
gls_test = gls_dummy[gls_dummy.year_launch >= time_threshold]
x = gls_train.drop(target, axis=1)
y = gls_train[target]
x_test_to_select = gls_test.drop(target, axis=1)
y_test_to_select = gls_test[target]
# dmat = DMatrix(data=x, label=y)

x_train_to_append, x_test, y_train_to_append, y_test = train_test_split(x_test_to_select, y_test_to_select, test_size=0.5, random_state=42)
x_train = pd.concat([x, x_train_to_append])
y_train = pd.concat([y, y_train_to_append])
test_size = round(len(x_test) / len(gls), 2)

# initial model with default params
# # train-test split
# test_size = 0.2
# x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=test_size)
train_data, test_data = DMatrix(x_train, label=y_train), \
                        DMatrix(x_test, label=y_test)

# initialize xgboost and record result
xgb = XGBRegressor(objective='reg:squarederror').fit(x_train, y_train)
pred_train, pred_test = xgb.predict(x_train), \
                        xgb.predict(x_test)
mape_train, mape_test = mape(y_train, pred_train), \
                        mape(y_test, pred_test)
res_df = pd.DataFrame({'mape_train': [mape_train],
                       'mape_test': [mape_test]}
                      )
key_params = ['max_depth',
              'learning_rate',
              'gamma',
              'reg_lambda',
              'min_child_weight'
              ]
initial_params = xgb.get_params()
for param in key_params:
    res_df[param] = initial_params[param]

# random search cv
# define search space
param_space = {'max_depth': np.arange(3, 10),
               'learning_rate': np.arange(0.01, 0.3, 0.01),
               'gamma': np.arange(0, 1, 0.05),
               'reg_lambda': np.arange(0.1, 5, 0.1),
               'min_child_weight': np.arange(1, 10),
               'subsample': np.arange(0.5, 1, 0.1),
               'colsample_bytree': np.arange(0.5, 1, 0.01),
               }

# random_search_output = tune_param_rs(x_train, y_train, x_test, y_test, test_size, params=param_space)
# random_search_res = random_search_output[0]
# mape_train, mape_test = random_search_output[1], random_search_output[2]
#
# print("Test size: %f" % test_size,
#       "MAPE train: %f" % mape_train,
#       "MAPE test: %f" % mape_test,
#       "MAPE test-train: %f" % (mape_test - mape_train),
#       sep='\n')

# # cross validation
# eval_res = cv(dtrain=train_data,
#               params=param_tuned,
#               nfold=3,
#               num_boost_round=100,
#               early_stopping_rounds=10,
#               metrics='mape',
#               as_pandas=True,
#               seed=42)

check = 42

param_tuned = {'subsample': 0.7,
               'reg_lambda': 0.8,
               'min_child_weight': 8,
               'max_depth': 5,
               'learning_rate': 0.02,
               'gamma': 0.45,
               'colsample_bytree': 0.92
               }

# test for over-fitting
xgb_test = train(params=param_tuned, dtrain=train_data, num_boost_round=500)
pred_train, pred_test = xgb_test.predict(DMatrix(x_train)), \
                        xgb_test.predict(DMatrix(x_test))
mape_train, mape_test = mape(y_train, pred_train), \
                        mape(y_test, pred_test)

print("Test size: %f" % test_size,
      "MAPE train: %f" % mape_train,
      "MAPE test: %f" % mape_test,
      "MAPE test-train: %f" % (mape_test - mape_train),
      sep='\n')

# prediction
feat = x.columns

for col in [item for item in feat if item not in pred.columns]:
    pred[col] = np.nan
pred_x = pd.get_dummies(pred[feature_cols])

# fill in dynamic features manually
pred_land_index_dict = {}
pred_index_list = pred.index
for i in range(len(pred)):
    idx = pred_index_list[i]
    pred_land_index_dict[pred.loc[idx, 'land_parcel_std']] = idx

dynamic_var = ['num_bidders', 'proj_max_floor', 'joint_venture']
pred_x.loc[pred_land_index_dict['Marina Gardens Lane'], 'zone'] = 'downtown core'
pred_x.loc[pred_land_index_dict['Lentor Gardens'], 'comparable_price_psm_gfa'] = 11882.8  # do not manually key in, find ways to auto-calculate
pred_x.loc[pred_land_index_dict['Marina Gardens Lane'], 'proj_max_floor'] = 40
pred_x.loc[pred_land_index_dict['Lentor Gardens'], 'proj_max_floor'] = 18
pred_x['joint_venture'] = 0


# dmat_pred = DMatrix(x, label=y)
dmat_pred = train_data
xgb_tuned = train(params=param_tuned, dtrain=dmat_pred, num_boost_round=500)
pred_train = xgb_tuned.predict(dmat_pred)
pred_test = xgb_tuned.predict(test_data)

for col in [item for item in xgb_tuned.feature_names if item not in pred_x.columns]:
    pred_x[col] = np.nan

pred_x = pred_x[xgb_tuned.feature_names]
pred_x_dmat = DMatrix(pred_x)
hi = gls[gls.year_launch == 2022].hi_price_psm_gfa.mean()
prediction = xgb_tuned.predict(pred_x_dmat)
mape = mape(y_test, pred_test)

for i in range(len(pred_index_list)):
    parcel_idx = pred_index_list[i]
    print("Predicted tender price for {}: {:.2f}".format(pred.loc[parcel_idx, 'land_parcel_std'],
                                                         pred_x.loc[parcel_idx, 'site_area_sqm'] * pred_x.loc[parcel_idx, 'gpr'] *
                                                         prediction[i]))

# for marina: 880,000,000 (MTD) / 1,180,000,000 (Yuelin)
# for lentor: 520,000,000

fig, ax = plt.subplots(figsize=(10, 10))
plot_importance(xgb_tuned, max_num_features=20, height=0.5, ax=ax, importance_type='gain')
plt.show()

check = 42
