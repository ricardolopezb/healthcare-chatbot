import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from nlp import symptoms, word_extractor, noun_token_extractor
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


training = pd.read_csv('Data/Training.csv') #Training csv
testing= pd.read_csv('Data/Testing.csv') #Testing csv
cols= training.columns #Columns for testing
cols= cols[:-1] # elimina la ultimacolumna delmarreglo de columnas pq es el tag de las enfermedades
x = training[cols] # contiene todas las columnas menos la que tiene el tag de las enfermedades
y = training['prognosis'] # contiene la 
y1= y


reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42) #Separa el dataset en train y test con un seed de 42
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy) #transforma los tags a valores numericos


clf1  = DecisionTreeClassifier() # lo hace el decision tree con los datos de train
clf = clf1.fit(x_train,y_train) # le mete los datos
# print(clf.score(x_train,y_train))
# print ("cross result========")
scores = cross_val_score(clf, x_test, y_test, cv=3) #lo testea con el train
# print (scores)
print (scores.mean()) # muestra la media

model=SVC() # hace lo mismo con svm
model.fit(x_train,y_train) # trainea el modelo
print("for svm: ")
print(model.score(x_test,y_test)) # lo testea

#sortea las features importantes del modelo para ver que onda
importances = clf.feature_importances_ 
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

#matchea el sintoma con su respectiva columna
for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

#metodo que hace el calculo de la severidad en funcion de los dias que se tiene el sintoma
def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")

# matche ala descripcion con el sintoma
def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)



# lee el csv de severidad y lo mete en un diccionario
# lo mapea con su indice de severidad segun el modelo
def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

# lee el csv de precauciones y lo mete en un diccionario
# es un mapa que te dice que hacer en cada caso de enfermedad que te detecte la ai
def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t",end="->")
    name=input("")
    print("Hello, ",name)

# metodo que hace el match de los sintomas con las enfermedades y los agrega a las posible diseases list
def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
    
# metodo que hace la prediccion de la enfermedad en base a los sintomas que le das
def sec_predict(symptoms_exp):
    # entrena el modelo con los datos de train
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    #detecta los sintomas que le pasaste y los mete en un vector
    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    #crea un arreglo de 0 y lo llena con 1 en los indices de los sintomas que le pasaste
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1
    #devuelve la prediccion del modelo
    return rf_clf.predict([input_vector])

#dado el nodo agarra la disease y la imprime
def print_disease(node):
    node = node[0]
    val  = node.nonzero()  # agarra los valores que no son 0 del nodo
    disease = le.inverse_transform(val[0]) # le hace el inverse transform para que te devuelva el nombre de la enfermedad
    return list(map(lambda x:x.strip(),list(disease))) # devuelve la enfermedad, borra los espacios en blanco de las enfermedades para devolver un string decente

# metodo que hace el arbol de decision
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []
    while True:

        print("\nEnter the symptom you are experiencing  \t\t",end="->")
        disease_input = input("")
        #TODO: aca hay que poner el intercepter de ortografia
        #checks the pattern of the input given by the user
        found_symptoms = symptoms(noun_token_extractor(disease_input))
        print("FOUND SYMPTOMS", found_symptoms)
        conf,cnf_dis=check_pattern(chk_dis, found_symptoms[0]) #agarra el input y lo matchea con las enfermedades devuelve 0,[] si no matchea, devuelve 1 y la lista d einputs si matchea alguno
        if conf==1:
            print("searches related to input: ")
            for num,it in enumerate(cnf_dis):
                print(num,")",it)
            if num!=0:
                #pregunta si hay mas de un sintoma matcheado cual es y que de una ompcion entre 0 y la cantidad de sintomas
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp=0

            disease_input=cnf_dis[conf_inp]
            break
            # print("Did you mean: ",cnf_dis,"?(yes/no) :",end="")
            # conf_inp = input("")
            # if(conf_inp=="yes"):
            #     break
        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days=int(input("Okay. From how many days ? : "))
            break
        except:
            print("Enter valid input.")
    def recurse(node, depth):
        #agarra el primer nodo
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node] #agarra el nombre del nodo
            threshold = tree_.threshold[node] # agarra el threshold del nodo
            #tresh es 0.5 pq es un arbol binario
            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold: # si matchea con el sintoma va a la derecha y sino a la izquierda del arbol
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1) # si matchea con el sintoma va a la derecha y sino a la izquierda del arbol
        else:
            #en caso de que el nodo sea una hoja agarra la enfermedad y la imprime
            present_disease = print_disease(tree_.value[node]) #agarra la posible enfermedad y la imprime
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            # dis_list=list(symptoms_present)
            # if len(dis_list)!=0:
            #     print("symptoms present  " + str(list(symptoms_present)))
            #print("symptoms given "  +  str(list(symptoms_given)) )
            print("Are you experiencing any ")
            symptoms_exp=[]
            for syms in list(symptoms_given): #Con al enfermedad posible agarra los sintomas y les pregunta si tiene alguno, si los tiene los apendea a la lista
                inp=""
                print(syms,"? : ",end='')
                while True:
                    inp=input("")
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ",end="")
                if(inp=="yes"):
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp) #hace una segunda prediccion con los sintomas que le pasaste
            # print(second_prediction)
            calc_condition(symptoms_exp,num_days) #calcula la condiciones de la enfermedad en base a los dias que le pasaste, y te dice si tenes que ver un doctor o no
            #si las dos matchea, imprime una, sino als dos condiciones, te dice que son y que hacer y si es grave si hay que ver al doctor o no
            if(present_disease[0]==second_prediction[0]):
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])

                #readn(f"You may have {present_disease[0]}")
                #eadn(f"{description_list[present_disease[0]]}")

            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            # print(description_list[present_disease[0]])
            precution_list=precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)

            # confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
            # print("confidence level is " + str(confidence_level))

    recurse(0, 1)
getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
tree_to_code(clf,cols)
print("----------------------------------------------------------------------------------------")

