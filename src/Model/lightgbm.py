import lightgbm as lgb
import matplotlib.pyplot as plt

def LightGBM(X_train, X_test, y_train, y_test, fig):
    lgb_results = {}                                    # 学習の履歴を入れる入物
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    params = {
          'task': 'train',              # タスクを訓練に設定
          'boosting_type': 'gbdt',      # GBDTを指定
          'objective': 'regression',    # 回帰を指定
          'metric': 'rmse',             # 回帰の評価関数
          'learning_rate': 0.1,         # 学習率
          'feature_fraction': 0.8,
          'num_leaves': 63
          }
    
    model = lgb.train(
                      params=params,                    # ハイパーパラメータをセット
                      train_set=lgb_train,              # 訓練データを訓練用にセット
                      valid_sets=[lgb_train, lgb_test], # 訓練データとテストデータをセット
                      valid_names=['Train', 'Test'],    # データセットの名前をそれぞれ設定
                      num_boost_round=200,              # 計算回数
                      early_stopping_rounds=50,         # アーリーストッピング設定
                      evals_result=lgb_results,             # 学習の履歴を保存
                      verbose_eval=-1                           # ログを最後の1つだけ表示
                      )  
    
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
    
        

    