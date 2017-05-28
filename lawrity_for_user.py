# -*- coding: cp1251 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import coverage_error
from sklearn.metrics import hamming_loss
from sklearn.metrics import label_ranking_average_precision_score
from tkinter import *
from skmultilearn.adapt.mlknn import MLkNN
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
import time

def tranformToBin(y):
    for i in range(0,y.size):
        y[i]=y[i].split(',')
    y_int = [map(int, x) for x in y]
    y_new = []
    for row in y_int:
        tmp = [0]*71
        for el in row:
            tmp[el] = 1
        y_new.append(tmp)
    return y_new

#UserApi

import tkFileDialog

def Quit(ev):
    global root
    root.destroy()
    
def askDirectory(self):
    """Returns a selected directoryname."""
    textbox.configure(state='normal')
    filename = tkFileDialog.askopenfilename(filetypes=[('*.csv files','csv')])
    if filename == '':
        return
    textbox.delete('1.0', 'end') 
    textbox.insert('1.0', filename)
    textbox.configure(state='disabled')
    file_var.set(filename)
    return filename

def start(ev):
    resultbox.configure(state='normal')
    resultbox.delete('1.0', END)
    training_set = pd.read_csv(file_var.get(), sep = ';')
    training_set.info()

    xcols = [col for col in training_set.columns if col != 'Class']

    X = training_set[xcols].values
    y = training_set['Class'].values

    X = np.array(X, dtype=np.float64)
    y_new = np.array(tranformToBin(y), dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y_new, test_size = 0.2, random_state=99)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    if(var.get() == 0):
        clf = OneVsRestClassifier(LinearSVC())
    else:
        lst = []
        lst_of_ac = []
        k_opt = 0
        for k in range(1,15):
            knn = MLkNN(k,s=1.0)
            scores = cross_val_score(knn, X, y_new, cv=cv, scoring='accuracy')
            lst_of_ac.append(scores.mean())
            lst.append(str(k))
        cnt = 0
        for i in lst_of_ac:
            if i == max(lst_of_ac):
                k_opt = cnt
            cnt += 1
        clf = MLkNN(k_opt,s=1.0)                
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    start = time.time()
    precision_recall_fscore_support(y_test, y_pred, average='samples')
    ac = cross_val_score(clf, X, y_new, cv=cv, scoring='accuracy')
    pr = cross_val_score(clf, X, y_new, cv=cv, scoring='precision_samples')
    re = cross_val_score(clf, X, y_new, cv=cv, scoring='recall_samples')
    f1 = cross_val_score(clf, X, y_new, cv=cv, scoring='f1_samples')
    f1_micro = cross_val_score(clf, X, y_new, cv=cv, scoring = 'f1_micro')
    f1_macro = cross_val_score(clf, X, y_new, cv=cv, scoring = 'f1_macro')
    precision_micro = cross_val_score(clf, X, y_new, cv=cv, scoring = 'precision_micro')
    precision_macro = cross_val_score(clf, X, y_new, cv=cv, scoring = 'precision_macro')
    recall_micro = cross_val_score(clf, X, y_new, cv=cv, scoring = 'recall_micro')
    recall_macro = cross_val_score(clf, X, y_new, cv=cv, scoring = 'recall_macro')
    end = time.time()
    if(var.get() == 0):
         resultbox.insert('1.0','Results:\n')
    else:
         resultbox.insert('1.0','Results:\n Optimal k: ' + str(k_opt) + '\n')

    resultbox.insert('3.0','Accuracy: '
                       +str(ac.mean())
                       +'\n Precision: '
                       +str(pr.mean())
                       +'\n Recall: '
                       +str(re.mean())
                       +'\n F_1 measure: '
                       +str(f1.mean())
                       +'\n F_1_micro measure: '
                       +str(f1_micro.mean())
                       +'\n F_1_macro measure: '
                       +str(f1_macro.mean())
                       +'\n Precision_micro measure: '
                       +str(precision_micro.mean())
                       +'\n Precision_macro measure: '
                       +str(precision_macro.mean())
                       +'\n Recall_micro measure: '
                       +str(recall_micro.mean())
                       +'\n Recall_macro measure: '
                       +str(recall_macro.mean())   
                       +'\n Hamming Loss: '
                       +str(hamming_loss(y_test, y_pred))
                       +'\n Zero One Loss: '
                       +str(zero_one_loss(y_test, y_pred))
                       +'\n Time: '
                       +str(end-start)
                     )
    resultbox.configure(state='disabled')
    
    
root = Tk()
root.resizable(width=False, height=False)
root.minsize(width = 400, height = 600)
var=IntVar()
var.set(1)
file_var = StringVar()

panelFrame = Frame(root, height = 500, bg = 'gray')
textFrame = Frame(root, height = 1, width = 40)

panelFrame.pack(side = 'bottom', fill = 'both')
textFrame.pack(side = 'top', fill = 'none', expand = 1)

textbox = Text(textFrame, font='TimesNewRoman 8', wrap='word', height = 1, width = 40, borderwidth = 2)
textbox.pack(side = 'left', fill = 'both', expand = 1)

resultbox = Text(panelFrame, font='TimesNewRoman 12', wrap='word', height = 100, width = 40, borderwidth = 2)

resultbox.pack(side = 'left', fill = 'both', expand = 1)
resultbox.configure(state='disabled')



loadBtn = Button(panelFrame, text = 'Load')
quitBtn = Button(panelFrame, text = 'Quit')
startBtn = Button(panelFrame, text = 'Start')


rad0 = Radiobutton(panelFrame,text="Binary Relevance with SVM",
          variable=var,value=0)
rad1 = Radiobutton(panelFrame,text="ML-kNN",
          variable=var,value=1)
rad0.pack()
rad1.pack()

loadBtn.bind("<Button-1>", askDirectory)
quitBtn.bind("<Button-1>", Quit)
startBtn.bind("<Button-1>", start)

loadBtn.place(x = 10, y = 10, width = 40, height = 40)
quitBtn.place(x = 60, y = 10, width = 40, height = 40)
rad0.place (x = 120, y = 10)
rad1.place (x=120, y=40)


startBtn.place(x = 20, y = 60, width = 60, height = 40)

resultbox.place(x = 10, y = 140)


root.mainloop()
