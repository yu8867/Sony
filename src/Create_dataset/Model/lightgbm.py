import lightgbm as lgb
import matplotlib.pyplot as plt

def LightGBM(X_train, X_test, y_train, y_test, fig):
    lgb_results = {}
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test)
    
    params = {'task': 'train',
              'boosting_type': 'gbdt',
              'objective': 'regression',
              'metric': 'rmse',
              'learning_rate': 0.1,
              'random_seed': 123,
              }
    
    model = lgb.train(params=params,
                      train_set=lgb_train,
                      valid_sets=[lgb_train,lgb_test],
                      valid_names=['Train', 'Test'],
                      num_boost_round=100,
                      early_stopping_rounds=50,
                      evals_result=lgb_results,
                      verbose_eval=-1)
    
    loss_train = lgb_results['Train']['rmse']
    loss_test = lgb_results['Test']['rmse']
    
    if fig==1:
        plt.plot(loss_train, label='train_loss')
        plt.plot(loss_test, label='test_loss')
        plt.xlabel('iteration')
        plt.ylabel('rmse')
        plt.legend()
        plt.show()
        
    return model
    
        

    