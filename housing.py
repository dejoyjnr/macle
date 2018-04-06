import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder, Imputer, RobustScaler
from xgboost import XGBRegressor
from sklearn.base import TransformerMixin, BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars, LassoCV, LassoLarsCV
from sklearn.linear_model import OrthogonalMatchingPursuit, HuberRegressor, RANSACRegressor, LassoLarsIC
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import skew
import operator, datetime
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
from scipy.special import boxcox1p

lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

model_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

models = [LinearRegression(),
          Ridge(),
          ElasticNet(),
          Lasso(),
          # Lars(),
          LassoLars(),
          BayesianRidge(),
          HuberRegressor(),
          RANSACRegressor(),
          OrthogonalMatchingPursuit(),
          DecisionTreeRegressor(),
          RandomForestRegressor(),
          ExtraTreesRegressor(),
          KNeighborsRegressor(),
          GradientBoostingRegressor(),
          XGBRegressor(),
          SVR(),
          MLPRegressor()
          ]

def housing():

    class DataFrameImputer(TransformerMixin):

        def __init__(self):
            """Impute missing values.
            Columns of dtype object are imputed with the most frequent value
            in column.
            Columns of other types are imputed with mean of column.
            """
        def fit(self, X):

            self.fill = pd.Series([X[c].value_counts().index[0]
                if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                index=X.columns)
            return self

        def transform(self, X):
            return X.fillna(self.fill)

    # Read the train and test dataset
    file_path ='C:/Users/amankwad/PycharmProjects/macle/data/'
    # file_path = '/home/dejoy/.kaggle/competitions/house-prices-advanced-regression-techniques/'
    train = pd.read_csv(file_path+'train.csv')
    test = pd.read_csv(file_path+'test.csv')

    ############################################################################################################################

    # # Select features and target
    # y = train.SalePrice
    # X = train.drop(['SalePrice', 'Id'], axis=1)
    #
    # k=train.isnull().sum()
    # k=k.iloc[train.isnull().sum().nonzero()]
    # to_be_removed_cols = []
    # for key,value in k.iteritems():
    #     if value >= 0.65*train.shape[0]:
    #         to_be_removed_cols.append(key)
    #
    # cleaned_X = X.drop(to_be_removed_cols, axis=1)
    # cols_non_object = cleaned_X.select_dtypes(exclude=['object']).columns.tolist()
    # cols_object = cleaned_X.select_dtypes(include=['object']).columns.tolist()
    #
    # imputed_X = DataFrameImputer().fit_transform(cleaned_X)
    #
    # encoded_X = imputed_X.apply(LabelEncoder().fit_transform)
    # df = encoded_X[cols_non_object].copy()
    # df['SalePrice'] = y
    # pc = df.corr()
    #
    # pcs = pc.loc['SalePrice', :]
    # preds = []
    # for key,value in pcs.iteritems():
    #     if abs(value) >= 0.05:
    #         preds.append(key)
    # preds = preds[:-1]
    # preds_no_objects = preds.copy()
    # for c in cols_object:
    #     preds.append(c)
    #
    # models_names = ['KNeighborsRegressor', 'LinearRegression', 'GradientBoostingRegressor', 'XGBRegressor', 'Ridge', 'Lasso',
    #          'ElasticNet', 'Lars', 'LassoLars', 'BayesianRidge', 'HuberRegressor',
    #           'RANSACRegressor']
    #
    # # print(model_finder(encoded_X[preds], y))
    #
    # # preds1 = []
    # # for k, v in re.items():
    # #     if abs(v[1]) >= 0.1:
    # #         preds1.append(k)
    #
    # print(np.sqrt(-1*cross_val_score(XGBRegressor(n_estimators=2950, learning_rate=0.017, n_jobs=4), encoded_X[preds], y,
    #                          scoring = 'neg_mean_squared_log_error', cv=5).mean()))
    # print(np.sqrt(-1*cross_val_score(GradientBoostingRegressor(n_estimators=1000, learning_rate=0.0454), encoded_X[preds], y,
    #                          scoring = 'neg_mean_squared_log_error', cv=5).mean()))
    # # 0.1239001233620085
    # # 0.12461888447859339
    #
    # # cleaned_test = test.drop(to_be_removed_cols, axis=1)
    # # imputed_test = DataFrameImputer().fit_transform(cleaned_test)
    # # encoded_test = imputed_test.apply(LabelEncoder().fit_transform)
    # #
    # #
    # # model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.0454)
    # # model.fit(encoded_X[preds], y)
    # # predictions = model.predict(encoded_test[preds])
    # #
    # # my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})
    # # print(my_submission.sample(10))
    # # my_submission.to_csv('sub.csv', index=False)

    ############################################################################################################################


    def mean_squared_error_(ground_truth, predictions):
        return mean_squared_error(ground_truth, predictions) ** 0.5


    RMSE = make_scorer(mean_squared_error_, greater_is_better=False)

    def create_submission(prediction, score):
        now = datetime.datetime.now()
        sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
        # sub_file = 'prediction_training.csv'
        print('Creating submission: ', sub_file)
        pd.DataFrame({'Id': test['Id'].values, 'SalePrice': prediction}).to_csv(sub_file, index=False)


    def data_preprocess(train, test):
        # outlier_idx = [4, 11, 13, 20, 46, 66, 70, 167, 178, 185, 199, 224, 261, 309, 313, 318, 349, 412, 423, 440, 454, 477, 478, 523, 540, 581, 588, 595,
        #                654, 688, 691, 774, 798, 875, 898, 926, 970, 987, 1027, 1109, 1169, 1182, 1239, 1256, 1298, 1324, 1353, 1359, 1405, 1442, 1447]
        # train.drop(train.index[outlier_idx], inplace=True)
        all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                              test.loc[:, 'MSSubClass':'SaleCondition']))

        to_delete = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
        all_data = all_data.drop(to_delete, axis=1)

        train["SalePrice"] = np.log1p(train["SalePrice"])
        # log transform skewed numeric features
        numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
        skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
        skewed_feats = skewed_feats[skewed_feats > 0.45]
        skewed_feats = skewed_feats.index
        all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
        all_data = pd.get_dummies(all_data)
        all_data = all_data.fillna(all_data.mean())
        X_train = all_data[:train.shape[0]]
        X_test = all_data[train.shape[0]:]
        y_train = train.SalePrice

        return X_train, X_test, y_train

    def preprocess_data(train, test):
        outlier_idx = [4, 11, 13, 20, 46, 66, 70, 167, 178, 185, 199, 224, 261, 309, 313, 318, 349, 412, 423, 440, 454, 477, 478, 523, 540, 581, 588, 595,
                       654, 688, 691, 774, 798, 875, 898, 926, 970, 987, 1027, 1109, 1169, 1182, 1239, 1256, 1298, 1324, 1353, 1359, 1405, 1442, 1447]
        train.drop(train.index[outlier_idx], inplace=True)
        train["SalePrice"] = np.log1p(train["SalePrice"])
        all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                              test.loc[:, 'MSSubClass':'SaleCondition']))
        all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
        all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
        all_data["Alley"] = all_data["Alley"].fillna("None")
        all_data["Fence"] = all_data["Fence"].fillna("None")
        all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
        all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median()))
        for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
            all_data[col] = all_data[col].fillna('None')
        for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
            all_data[col] = all_data[col].fillna(0)
        for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
            all_data[col] = all_data[col].fillna(0)
        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            all_data[col] = all_data[col].fillna('None')
        all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
        all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
        all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
        all_data = all_data.drop(['Utilities'], axis=1)
        all_data["Functional"] = all_data["Functional"].fillna("Typ")
        all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
        all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
        all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
        all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
        all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
        all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
        all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
        all_data['OverallCond'] = all_data['OverallCond'].astype(str)
        all_data['YrSold'] = all_data['YrSold'].astype(str)
        all_data['MoSold'] = all_data['MoSold'].astype(str)

        cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
                'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
                'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
                'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
                'YrSold', 'MoSold')
        for c in cols:
            lbl = LabelEncoder()
            lbl.fit(list(all_data[c].values))
            all_data[c] = lbl.transform(list(all_data[c].values))
        all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
        numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
        skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew': skewed_feats})
        skewness = skewness[abs(skewness) > 0.75]
        skewed_features = skewness.index
        lam = 0.15
        for feat in skewed_features:
            # all_data[feat] += 1
            all_data[feat] = boxcox1p(all_data[feat], lam)
        all_data = pd.get_dummies(all_data)
        X_train = all_data[:train.shape[0]]
        X_test = all_data[train.shape[0]:]
        y_train = train.SalePrice

        return X_train, X_test, y_train



    def model_random_forecast(Xtrain, Xtest, ytrain):
        X_train = Xtrain
        y_train = ytrain
        rfr = RandomForestRegressor(n_jobs=1, random_state=0)
        param_grid = {} #'n_estimators': [500], 'max_features': [10,15,20,25], 'max_depth':[3,5,7,9,11]}
        model = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
        model.fit(X_train, y_train)
        print('Random forecast regression...')
        print('Best Params:')
        print(model.best_params_)
        print('Best CV Score:')
        print(-model.best_score_)

        y_pred = model.predict(Xtest)
        return y_pred, -model.best_score_


    def model_gradient_boosting_tree(Xtrain, Xtest, ytrain):
        X_train = Xtrain
        y_train = ytrain
        gbr = GradientBoostingRegressor(random_state=0)
        param_grid = {
            #       'n_estimators': [500],
            #       'max_features': [10,15],
            #	'max_depth': [6,8,10],
            #       'learning_rate': [0.05,0.1,0.15],
            #      'subsample': [0.8]
        }
        model = GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
        model.fit(X_train, y_train)
        print('Gradient boosted tree regression...')
        print('Best Params:')
        print(model.best_params_)
        print('Best CV Score:')
        print(-model.best_score_)

        y_pred = model.predict(Xtest)
        return y_pred, -model.best_score_


    def model_xgb_regression(Xtrain, Xtest, ytrain):
        X_train = Xtrain
        y_train = ytrain

        xgbreg = XGBRegressor(seed=0)
        param_grid = {
                   'n_estimators': [300, 600, 900, 1200],
                   'learning_rate': [ 0.02, 0.04, 0.06, 0.08, 0.1],
                   'max_depth': [ 7],
                   'subsample': [ 0.8],
                   'colsample_bytree': [0.8],
                    }
        model = GridSearchCV(estimator=xgbreg, param_grid=param_grid, n_jobs=1, cv=5, scoring=RMSE)
        model.fit(X_train, y_train)
        print('eXtreme Gradient Boosting regression...')
        print('Best Params:')
        print(model.best_params_)
        print('Best CV Score:')
        print(-model.best_score_)

        y_pred = model.predict(Xtest)
        return y_pred, -model.best_score_


    def model_extra_trees_regression(Xtrain, Xtest, ytrain):
        X_train = Xtrain
        y_train = ytrain

        etr = ExtraTreesRegressor(n_jobs=1, random_state=0)
        param_grid = {} # 'n_estimators': [500], 'max_features': [10,15,20]}
        model = GridSearchCV(estimator=etr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
        model.fit(X_train, y_train)
        print('Extra trees regression...')
        print('Best Params:')
        print(model.best_params_)
        print('Best CV Score:')
        print(-model.best_score_)

        y_pred = model.predict(Xtest)
        return y_pred, -model.best_score_


    def grid_searcher(Xtrain, Xtest, ytrain, estimator, param_grid=None):
        if param_grid is None: param_grid = {}
        model = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=1, cv=5, scoring=RMSE, verbose=5)
        model.fit(Xtrain, ytrain)
        print('Best Params:')
        print(model.best_params_)
        print('Best CV Score:')
        print(-model.best_score_)

        y_pred = model.predict(Xtest)
        return y_pred, -model.best_score_


    def untuned_model_finder(X_train, y_train, models, scoring, cv, threshold, small_is_good=True):
        all_models = {}
        good_models = {}
        for model in models:
            name = str(model.__class__)
            score = -cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv).mean()
            all_models[name[8:len(name)-2].split(".")[-1]] = score
            if small_is_good:
                if score <= threshold: good_models[name[8:len(name) - 2].split(".")[-1]] = score
            else:
                if score >= threshold: good_models[name[8:len(name) - 2].split(".")[-1]] = score

        return sorted(good_models.items(), key=operator.itemgetter(1)), all_models

    def print_dict(dd):
        for k, v in dd.items():
            print("{} : {}".format(k, v))

    class ensemble(object):
        def __init__(self, n_folds, stacker, base_models):
            self.n_folds = n_folds
            self.stacker = stacker
            self.base_models = base_models

        def fit_predict(self, train, test, ytr):
            X = train.values
            y = ytr.values
            T = test.values
            folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=0)
            S_train = np.zeros((X.shape[0], len(self.base_models)))
            S_test = np.zeros((T.shape[0], len(self.base_models)))
            for i, reg in enumerate(base_models):
                print("Fitting the base model...")
                S_test_i = np.zeros((T.shape[0], self.n_folds))
                for j, (train_idx, test_idx) in enumerate(folds.split(X)):
                    X_train = X[train_idx]
                    y_train = y[train_idx]
                    X_holdout = X[test_idx]
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X_holdout)[:]
                    S_train[test_idx, i] = y_pred
                    S_test_i[:, j] = reg.predict(T)[:]
                S_test[:, i] = S_test_i.mean(1)

            print("Stacking base models...")
            # tuning the stacker

            param_grid = {
                # 'alpha': [i*0.001 for i in range(1, 101, 2)]
            }
            grid = GridSearchCV(estimator=self.stacker, param_grid=param_grid, n_jobs=1, cv=5, scoring=RMSE)
            grid.fit(S_train, y)
            try:
                print('Param grid:')
                print(param_grid)
                print('Best Params:')
                print(grid.best_params_)
                print('Best CV Score:')
                print(-grid.best_score_)
                print('Best estimator:')
                print(grid.best_estimator_)
            except:
                pass

            y_pred = grid.predict(S_test)[:]
            return y_pred, -grid.best_score_

    # build a model library (can be improved)
    base_models = [
        RandomForestRegressor(
            n_jobs=1, random_state=0,
            n_estimators=500, max_features=14
        ),
        RandomForestRegressor(
            n_jobs=1, random_state=0,
            n_estimators=500, max_features=20,
            max_depth=7
        ),
        RandomForestRegressor(
            n_jobs=1, random_state=0,
            n_estimators=500, max_features='log2'
        ),
        RandomForestRegressor(
            n_jobs=1, random_state=0,
            n_estimators=500, max_features='sqrt',
            max_depth=7
        ),
        ExtraTreesRegressor(
            n_jobs=1, random_state=0,
            n_estimators=500, max_features=15
        ),
        ExtraTreesRegressor(
            n_jobs=1, random_state=0,
            n_estimators=500, max_features=20
        ),
        ExtraTreesRegressor(
            n_jobs=1, random_state=0,
            n_estimators=500, max_features='log2'
        ),
        ExtraTreesRegressor(
            n_jobs=1, random_state=0,
            n_estimators=500, max_features='sqrt'
        ),
        GradientBoostingRegressor(
            random_state=0,
            n_estimators=500, max_features=10, max_depth=6,
            learning_rate=0.05, subsample=0.8
        ),
        GradientBoostingRegressor(
            random_state=0,
            n_estimators=500, max_features=15, max_depth=6,
            learning_rate=0.05, subsample=0.8
        ),
        GradientBoostingRegressor(
            random_state=0,
            n_estimators=500, max_features='log2', max_depth=6,
            learning_rate=0.05, subsample=0.8
        ),
        GradientBoostingRegressor(
            random_state=0,
            n_estimators=500, max_features='sqrt', max_depth=6,
            learning_rate=0.05, subsample=0.8
        ),
        XGBRegressor(
            seed=0,
            n_estimators=500, max_depth=10,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.75
        ),

        XGBRegressor(
            seed=0,
            n_estimators=500, max_depth=7,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.75
        ),
        # KNeighborsRegressor(n_neighbors=5),
        # KNeighborsRegressor(n_neighbors=10),
        # KNeighborsRegressor(n_neighbors=15),
        # KNeighborsRegressor(n_neighbors=25),
        # LassoLarsCV(),
        ElasticNet(),
        SVR(),
        Ridge(),
        BayesianRidge(),
        OrthogonalMatchingPursuit(),
        # ENet,
        # GBoost,
        # KRR,
        # lasso,
        # model_lgb,
        # model_xgb
    ]
#############################################################################################################

    #
    # xgb = XGBRegressor(seed=0), \
    #                   {
    #                       'n_estimators': range(1000, 2100, 100),
    #                        'learning_rate': [ 0.02, 0.04, 0.06, 0.08, 0.1, 0.5, 0.8],
    #                        'max_depth': [ 7],
    #                        'subsample': [ 0.8],
    #                        'colsample_bytree': [0.8],
    #                   }
    #
    # gbr = GradientBoostingRegressor(random_state=0), \
    #                   {
    #                       'n_estimators': range(1000, 2100, 100),
    #                       'learning_rate': [0.02, 0.04, 0.06, 0.08, 0.1, 0.5, 0.8],
    #                       'max_depth': [7],
    #                       'subsample': [0.8],
    #                       'colsample_bytree': [0.8],
    #                   }

    X_train, X_test, y_train = preprocess_data(train, test)

    ensem = ensemble(n_folds=5, stacker=Ridge(), base_models=base_models)

    y_pred, score = ensem.fit_predict(X_train, X_test, y_train)

    class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
        def __init__(self, models):
            self.models = models

        # we define clones of the original models to fit the data in
        def fit(self, X, y):
            self.models_ = [clone(x) for x in self.models]

            # Train cloned base models
            for model in self.models_:
                model.fit(X, y)

            return self

        # Now we do the predictions for cloned models and average them
        def predict(self, X):
            predictions = np.column_stack([
                model.predict(X) for model in self.models_
            ])
            return np.mean(predictions, axis=1)

    averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))

    # print(-1 * cross_val_score(averaged_models, X_train, y_train,
    #                          scoring=RMSE, cv=5).mean())

    # res = untuned_model_finder(X_train, y_train, models, RMSE, 5, threshold=0.15)
    # for elem in res[0]:
    #     print(elem)
    # print("========================================================================")
    # print_dict(res[1])



    # grid_searcher(X_train, X_test, y_train, KNeighborsRegressor(), {'n_neighbors': range(1, 101, 1)})
    # grid_searcher(Xtrain, Xtest, ytrain, gbr[0], gbr[1])

    # test_predict, score = model_random_forecast(Xtrain,Xtest,ytrain)
    # test_predict, score = model_xgb_regression(Xtrain,Xtest,ytrain)
    # test_predict, score = model_extra_trees_regression(Xtrain, Xtest, ytrain)
    # test_predict, score = model_gradient_boosting_tree(Xtrain,Xtest,ytrain)

    # create_submission(np.exp(y_pred), score)



if __name__ == "__main__":
    housing()