import pandas as pd
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lifelines.utils import concordance_index
import os

data=pd.read_csv('C:/Users/Soorya/WFH/Code/ML_in_diseases/tumor_data.tsv',sep='\t')
data_set = data.loc[:,['patient_id','BRCA1','BRCA2','TP53','EGFR','PTEN']]
survival = pd.read_csv('C:/Users/Soorya/WFH/Code/ML_in_diseases/survival_data.tsv',sep='\t')
full = pd.merge(survival,data_set,left_on='case_submitter_id',right_on='patient_id')
full.drop_duplicates('case_submitter_id',inplace=True)
print("loaded and ready")

# split into input (X) and output (Y) variables
X = full.iloc[:,4:9]
Y = full.iloc[:,1]

X.set_index(full.iloc[:,0],inplace=True)
Y=pd.DataFrame(Y).set_index(full.iloc[:,0])

X_train_full,X_test,Y_train_full,Y_test = train_test_split(X,Y,test_size=0.33, random_state=42)
X_train,X_valid,y_train,y_valid = train_test_split(X_train_full,Y_train_full,test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.fit_transform(X_valid)
X_test_scaled = scaler.fit_transform(X_test)

model = Sequential([
    Dense(5,activation="relu",input_shape=X_train.shape[1:]),
    Dense(3,activation="relu"),
    Dense(1)
    ])

model.compile(loss="mean_squared_error",optimizer='adam')
history = model.fit(X_train,y_train,epochs=30,validation_data=(X_valid,y_valid))
mse_test = model.evaluate(X_test,Y_test)
y_pred = model.predict(X_test)
full.set_index('case_submitter_id',drop=True,inplace=True)
print(concordance_index(Y_test,y_pred,full.loc[Y_test.index,'event']))

root_logdir = os.path.join(os.curdir,"my_logs")
def get_run_logdir(): 
    import time 
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S") 
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb])