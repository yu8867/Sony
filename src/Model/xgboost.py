import xgboost as xgb

def XGBOOST(X_train, X_test, y_train, y_test, fig):
    xgb_results = {}
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate':0.01,
        'max_depth':6,
    }
    
    model = xgb.train(params,
                      dtrain,
                      num_boost_round=10000,
                      evals=evals,
                      evals_result=xgb_results,
                      early_stopping_rounds=20,
                      verbose_eval=200
                      )
    
    return model
    
    
