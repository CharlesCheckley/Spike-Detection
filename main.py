import scipy.io as spio
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.signal import butter, lfilter
import scipy.signal
import pandas as pd
from sklearn.model_selection import train_test_split
from  customModel import CustomMLP
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from SimulatedAnnealing import annealing 


#butter bandpass filter for filtering the input data 
def butter_bandpass_filter(data, lowcut = 30 , highcut = 2500, fs = 25000, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    output = lfilter(b, a, data)
    return output

#loads the matlab training datafile
def load_training_data():
    mat = spio.loadmat('C:\\Users\\charl\\OneDrive - University of Bath\\Year 4\\CI\\Coursework C\\training.mat', squeeze_me=True)
    Index = mat['Index']
    Class = mat['Class']
    d = mat['d']
    return d, Index, Class 

#loads the matlab training datafile
def load_submission_data():
    mat = spio.loadmat('C:\\Users\\charl\\OneDrive - University of Bath\\Year 4\\CI\\Coursework C\\submission.mat', squeeze_me=True)
    d = mat['d']
    return d


#Returns data points of the spikes. 
#Data is stored in rows. If a class array is provided then the
# first item in a row is the class of the spike
def get_spikes_data(d, Index, Class = None): 
    dataset = []
    for i in range(len(Index)):
        spike = d[Index[i]:Index[i]+50]
        spike = spike.tolist()
        if type(Class) is np.ndarray:
            classification = Class[i]
            row = [classification] + spike
        else: 
            row = spike
        dataset.append(row) 
    dataset = pd.DataFrame(dataset)
    return dataset 


#splits the peaks for training and validation 
def split_spikes_data(data, test_size = 0.4):
    y = data.iloc[:,0]
    X = data.iloc[:,1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state=0)
    return X_train, X_test, y_train, y_test


#find the spikes in a dataset 
def find_spikes(data): 
    noise_mad = np.median(np.absolute(data)) / 0.6745
    peaks, _ = scipy.signal.find_peaks(data, height= 5*noise_mad, distance = 10)
    #find peaks finds the highest point of the spike.
    #In the prelabelled dataset the spikes are labelled at their base 
    #therefore the spikes are set as the peak value - 10. 
    spikes = peaks - 10 
    #if -10 causes a negative index value set that vaule to 0. 
    spikes = np.where(spikes<0, 0, spikes)
    return spikes


def score_spikes(spikes, Index, filtered_data, data, plot=False):
    spikes_detected = len(spikes)
    print(spikes_detected, "spikes were detected")
    print("There were actually", len(Index), "spikes")
    #calculate which spikes were correctly found
    Index = np.sort(Index)
    #array of 0s
    correct_Index = np.zeros(len(Index), dtype=bool) 
    correct_spikes = []
    for idx, val in enumerate(Index): 
            #find difference between the correct peak val and all other spikes 
            i = (np.abs(spikes - val)).argmin()
            #if difference of closest peak is less than 20 mark as correct  
            if abs(spikes[i] - Index[idx]) < 20: 
                correct_Index[idx] = 1
                correct_spikes.append(spikes[i])
                spikes = np.delete(spikes, i)
    incorrect_spikes = spikes
    print(len(correct_spikes), "spikes were correctly found")
    not_detected_spikes = np.delete(Index, correct_Index.nonzero())
    print(len(not_detected_spikes), " were not detected")
    print("and there were ", len(incorrect_spikes), "incorrect spikes")


    if plot: 
        fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, sharey=True) 
        
        #changes the x axis units to seconds rather than samples
        d_time = np.zeros((1440000,2))
        for i in range(len(data)):
            d_time[i, 0] = i / 25000
            d_time[i,1] = data[i]

        d_filt_time = np.zeros((1440000,2))
        for i in range(len(data)):
            d_filt_time[i, 0] = i / 25000
            d_filt_time[i,1] = filtered_data[i]

        ax1.plot(d_filt_time[:,0], d_filt_time[:,1])
        ax2.plot(d_time[:,0], d_time[:,1])


        #ax1 plots the predicted spikes and colours them correct or incorrect 
        for i in correct_spikes:
            ax1.axvline(x=i/25000, color='b', alpha=0.2)
        for i in incorrect_spikes:
            ax1.axvline(x=i/25000, color='r', alpha=0.2)


        #ax2 plots the index and shows which ones weren't detected
        for i in Index: 
            ax2.axvline(x=i/25000, color='g', alpha=0.2)
        for i in not_detected_spikes:
            ax2.axvline(x=i/25000, color='k', alpha=1)

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude(mV)")
        plt.show()

    return correct_spikes, incorrect_spikes, not_detected_spikes


#demonstrate the amplitude threshold detectors performance
def show_threshold(): 
    #fetch and filter trainig data
    d, Index, Class = load_training_data()
    filtered_d = butter_bandpass_filter(d)
    
    spks = find_spikes(filtered_d)
    correct_spikes, incorrect_spikes, not_detected_spikes = score_spikes(spks, Index, filtered_d, d, plot=True)
    

#demonstrate the k nearest neighbours performance on the training data 
def show_knearest(): 
    print("Demonstrating K nearest:")
    #fetch and filter training data
    d , Index, Class = load_training_data()
    d = butter_bandpass_filter(d)
    data = get_spikes_data(d, Index, Class)
    y = data.iloc[:,0]
    X = data.iloc[:,1:]
    #import K nearest Neighbors Classifier from sklearn 
    print("number of neighbours is 3, weights = distance")
    neigh = KNeighborsClassifier(n_neighbors=9, weights = 'distance')
    #perform 5 fold cross valdation on the whole dataset
    print("5 fold cross validation accurasy scores:")
    scores = cross_val_score(neigh, X, y, cv=5)
    print(scores)
    #split the dataset into 80% training 20% validation 
    print("Splitting training data into 80% training, 20% validation")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #train the model on the 80% training dataset
    neigh.fit(X_train, y_train)
    #predict using the model the classes of the remaining 20%
    predictions = neigh.predict(X_test)
    #calculate the confusion matrix and accurasy score accordingly
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    print("Accuracy:")
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy)


#demonstrate the MLPs performance on the training data 
def show_mlp(): 
    print("Demonstrating MLP nearest:")
    #fetch and filter training data
    d , Index, Class = load_training_data()
    d = butter_bandpass_filter(d)
    data = get_spikes_data(d, Index, Class)
    y = data.iloc[:,0]
    X = data.iloc[:,1:]
    #create an instance of CustomMLP
    print("Input nodes is 50, output nodes = 5, hidden nodes = 100, learning rate = 0.01, epochs = 5")
    mlp = CustomMLP(input_nodes = 50, output_nodes = 5, hidden_nodes= 46, learning_rate=0.014 ,epochs=8 )
    #perform 5 fold cross valdation on the whole dataset
    print("5 fold cross validation accurasy scores:")
    scores = cross_val_score(mlp, X, y, cv=5)
    print(scores)
    #split the dataset into 80% training 20% validation 
    print("Splitting training data into 80% training, 20% validation")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #train the model on the 80% training dataset
    mlp.fit(X_train, y_train)
    #predict using the model the classes of the remaining 20%
    predictions = mlp.predict(X_test)
    #calculate the confusion matrix and accurasy score accordingly
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:")
    print(accuracy)

#demonstrate simutlated annealing optimisation of the MLPs hyperparameters
def show_simulated_annealing(): 
    print("Demonstrating Simulated Annealing")
    d , Index, Class = load_training_data()
    d = butter_bandpass_filter(d)
    data = get_spikes_data(d, Index, Class)
    y = data.iloc[:,0]
    X = data.iloc[:,1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sima = annealing([100,0.01,20], 0.9, 3, 0.25, X_train, y_train, X_test, y_test)  
    best_solution, lowest_cost = sima.anneal()
    print("best solution is: ", best_solution)
    print("accuracy was: ", 1-lowest_cost)


#produce the submission file in .mat format
def submit(): 
    #train model on entire training dataset
    d , Index, Class = load_training_data()
    d = butter_bandpass_filter(d)
    data = get_spikes_data(d, Index, Class)
    y = data.iloc[:,0]
    X = data.iloc[:,1:]
    neigh = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
    neigh.fit(X, y)
    
    #load submission dataset, detect and classify spikes
    sub_d = load_submission_data()
    sub_d = butter_bandpass_filter(sub_d)
    Index = find_spikes(sub_d)
    print("spikes found", len(Index))
    spikes = get_spikes_data(sub_d, Index)
    #save Class and Index data in the expected format 
    Class = neigh.predict(spikes)
    Class = Class.astype(float)
    Index = Index.astype(float)
    submission_data = {}
    submission_data['Index'] = Index
    submission_data['Class'] = Class
    spio.savemat('C:\\Users\\charl\\OneDrive - University of Bath\\Year 4\\CI\\Coursework C\\13798.mat', submission_data)



#uncomment a function to run it 

if __name__ == "__main__":
    #show_mlp()
    #show_knearest()
    #show_simulated_annealing()
    #submit()
    show_threshold()
