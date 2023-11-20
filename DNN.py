#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Required Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv1D
import wfdb                            # Package for loading the ecg and annotation
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore") 
import random
from keras.layers import Bidirectional, LSTM
# Random Initialization
random.seed(42)


# In[2]:


# List of Patients
patients = ['100','101','102','103','104','105','106','107',
           '108','109','111','112','113','114','115','116',
           '117','118','119','121','122','123','124','200',
           '201','202','203','205','207','208','209','210',
           '212','213','214','215','217','219','220','221',
           '222','223','228','230','231','232','233','234']


# In[3]:


# Importing Data
data = r'C:\\Users\\Admin\\Desktop\\ECG\\mit-bih-arrhythmia-database-1.0.0\\'


# In[4]:


# Creating a Empty Dataframe
symbols_df = pd.DataFrame()

# Reading all .atr files 
for pts in patients:
    # Generating filepath for all .atr file names
    file = data + pts
    # Saving annotation object
    annotation = wfdb.rdann(file, 'atr')
    # Extracting symbols from the object
    sym = annotation.symbol
    # Saving value counts
    values, counts = np.unique(sym, return_counts=True)
    # Writing data points into dataframe
    df_sub = pd.DataFrame({'symbol': values, 'Counts': counts, 'Patient Number': [pts] * len(counts)})
    # Concatenating all data points  
    symbols_df = pd.concat([symbols_df, df_sub], axis=0, ignore_index=True)


# In[5]:


symbols_df


# In[6]:


# Value Counts of Different symbols in data
symbols_df.groupby('symbol').Counts.sum().sort_values(ascending = False)


# In[7]:


# Non Beat Symbols
nonbeat = ['[','!',']','x','(',')','p','t','u','`',
           '\'','^','|','~','+','s','T','*','D','=','"','@','Q','?']

# Abnormal Beat Symbols
abnormal = ['L','R','V','/','A','f','F','j','a','E','J','e','S']

# Normal Beat Symbols
normal = ['N']


# In[8]:


# Classifying normal, abnormal or nonbeat
symbols_df['category'] = -1
symbols_df.loc[symbols_df.symbol == 'N','category'] = 0
symbols_df.loc[symbols_df.symbol.isin(abnormal), 'category'] = 1


# In[9]:


# Value counts of different categories
value=symbols_df.groupby('category').Counts.sum()


# In[11]:


labels = ['Nonbeat ', 'Normal', 'Abnormal']
# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(value, labels=labels, autopct='%1.1f%%', startangle=140)

# Adding a title
plt.title('Distribution of Categories')

# Display the plot
plt.show()


# In[12]:


def load_ecg(file):    
    # load the ecg
    record = wfdb.rdrecord(file)
    # load the annotation
    annotation = wfdb.rdann(file, 'atr')
    
    # extracting the signal
    p_signal = record.p_signal

    # extracting symbols and annotation index
    atr_sym = annotation.symbol
    atr_sample = annotation.sample
    
    return p_signal, atr_sym, atr_sample


# In[13]:


# Accessing the ecg points for 
file = data + patients[8]


# In[14]:


# Accessing the load ECG function and getting annotation.symbol, annotation.sample, signals
p_signal, atr_sym, atr_sample = load_ecg(file)


# In[15]:


# Analysing annotations value counts for a single record
values, counts = np.unique(sym, return_counts=True)
for v,c in zip(values, counts):
    print(v,c)


# In[18]:


def make_dataset(pts, num_sec, fs, abnormal):
    # function for making dataset ignoring non-beats
    # input:
    #   pts - list of patients
    #   num_sec = number of seconds to include before and after the beat
    #   fs = frequency
    # output: 
    #   X_all = signal (nbeats , num_sec * fs columns)
    #   Y_all = binary is abnormal (nbeats, 1)
    #   sym_all = beat annotation symbol (nbeats,1)
    
    # initialize numpy arrays
    num_cols = 2*num_sec * fs
    X_all = np.zeros((1,num_cols))
    Y_all = np.zeros((1,1))
    sym_all = []
    
    # list to keep track of number of beats across patients
    max_rows = []
    
    for pt in pts:
        file = data + pt
        
        p_signal, atr_sym, atr_sample = load_ecg(file)
        
        # grab the first signal
        p_signal = p_signal[:,0]
        
        # make df to exclude the nonbeats
        df_ann = pd.DataFrame({'atr_sym':atr_sym,
                              'atr_sample':atr_sample})
        df_ann = df_ann.loc[df_ann.atr_sym.isin(abnormal + ['N'])]
        
        X,Y,sym = build_XY(p_signal,df_ann, num_cols, abnormal)
        sym_all = sym_all+sym
        max_rows.append(X.shape[0])
        X_all = np.append(X_all,X,axis = 0)
        Y_all = np.append(Y_all,Y,axis = 0)
        
    # drop the first zero row
    X_all = X_all[1:,:]
    Y_all = Y_all[1:,:]

    return X_all, Y_all, sym_all


# In[19]:


def build_XY(p_signal, df_ann, num_cols, abnormal):
    # this function builds the X,Y matrices for each beat
    # it also returns the original symbols for Y
    
    num_rows = len(df_ann)

    X = np.zeros((num_rows, num_cols))
    Y = np.zeros((num_rows,1))
    sym = []
    
    # keep track of rows
    max_row = 0

    for atr_sample, atr_sym in zip(df_ann.atr_sample.values,df_ann.atr_sym.values):

        left = max([0,(atr_sample - num_sec*fs) ])
        right = min([len(p_signal),(atr_sample + num_sec*fs) ])
        x = p_signal[left: right]
        if len(x) == num_cols:
            X[max_row,:] = x
            Y[max_row,:] = int(atr_sym in abnormal)
            sym.append(atr_sym)
            max_row += 1
    X = X[:max_row,:]
    Y = Y[:max_row,:]
    return X,Y,sym


# In[20]:


# Parameter Values
num_sec = 3
fs = 360


# In[21]:


# Accessing the fuction and creating a dataset with ECG digital Points
X_all, Y_all, sym_all = make_dataset(patients, num_sec, fs, abnormal)


# In[22]:


X_all = pd.DataFrame(X_all)  # Your feature data
Y_all = pd.DataFrame(Y_all)  # Your label data
sym_all = pd.DataFrame(sym_all)  # Your symbol data

# Combine the three variables into a single dataset
combined_dataset = pd.concat([X_all, Y_all, sym_all], axis=1)


# In[23]:


combined_dataset


# In[24]:


combined_dataset.columns = [*combined_dataset.columns[:-2], 'Label', 'Symbols']


# In[25]:


combined_dataset = combined_dataset.drop('Symbols', axis=1)


# In[26]:


import matplotlib.pyplot as plt

# Assuming combined_dataset is your existing combined dataset
row_to_plot = combined_dataset.iloc[0]  # Replace '0' with the row index you want to plot

# Extract the label (if applicable)
label = row_to_plot['Label']  # Replace 'Label' with your label column name

# Extract the features
features = row_to_plot.drop('Label')  # Assuming 'Label' is the column with the label

# Convert feature index to string
feature_index_str = features.index.astype(str)

# Create a plot
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(feature_index_str, features)  # Convert feature index to string before plotting
plt.title(f'Row {0}, Label: {label}')
plt.xlabel('Time Index')
plt.ylabel('ECG Signals')

plt.show()


# In[27]:


combined_dataset


# In[28]:


combined_dataset['Label'].value_counts()


# In[29]:


combined_dataset.isna().sum()


# In[30]:


X = combined_dataset.drop(columns=['Label'])
y = combined_dataset[['Label']]


# In[31]:


# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Runing DNN

# In[32]:


# Relu for activation function and drop out for regularization
model = Sequential()
model.add(Dense(32, activation = 'relu', input_dim = X_train.shape[1]))
model.add(Dropout(rate = 0.25))
model.add(Dense(1, activation = 'sigmoid'))


# In[33]:


# Compiling model with  binary crossentropy and the adam optimizer
model.compile(loss = 'binary_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])


# In[55]:


# Fitting the model
model.fit(X_train, y_train, batch_size = 32, epochs= 10, verbose = 1)


# In[35]:


# Evaluation Metrics
def print_report(y_actual, y_pred, thresh):
    # Function to print evaluation metrics
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    specificity = sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)
    prevalence = (sum(y_actual)/len(y_actual))
    print('AUC:%.3f'%auc)
    print('Accuracy:%.3f'%accuracy)
    print('Recall:%.3f'%recall)
    print('Precision:%.3f'%precision)
    print('Specificity:%.3f'%specificity)
    print('Prevalence:%.3f'%prevalence)
    print(' ')
    return auc, accuracy, recall, precision, specificity


# In[37]:


# Predictions
y_train_preds_dense = model.predict(X_train,verbose = 1)
y_valid_preds_dense = model.predict(X_test,verbose = 1)


# In[52]:


# Convert the values in y_train to numeric type
numeric_y_train = np.array(y_train, dtype=np.float32)
numeric_y_test = np.array(y_test, dtype=np.float32)
# Threshold Value
thresh = (sum(numeric_y_train)/len(numeric_y_train))[0]


# In[53]:


# Accessing Evaluation Metrics Function
print('On Train Data')
print_report(numeric_y_train, y_train_preds_dense, thresh)
print('On test Data')
print_report(numeric_y_test, y_valid_preds_dense, thresh)


# In[ ]:




