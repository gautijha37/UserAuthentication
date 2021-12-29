# import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# import other scripts we wrote
from keystroke_feature_extractor import extract_keylog_features
from SOM import SOM

# Welcome prompt
print("="*100)
print("Welcome! This is the MIES Term project of Group 2")

print(" ___   ___  ___  _ _  ___    _  _ ")
print("/  _] | . \| . || | || . \  | || |")
print("| [_/\|   /| | || | ||  _/  | || |")
print("\____/|_\_\|___| \__||_|    |_||_|")
                                  

print("17EC34003\tManideep Mamindlapally \n17EC32004\tDigvijay Anand \n17EC32005\tGautam Jha \n17EC35036\tDebdut Mandal \n17EC35023\tShubham Somnath Sahoo")
print("We are using SOM dimensionality reduction to perform user authentication based on keystroke \ndynamics data.\n")

print("Before we proceed,  we'd like to thank the course instructor Prof. Sudipta Mukhopadhyay and the \nother teaching assistants who have given us this valuable opportunity and guided us throughout the \nsemester. Special thanks to TA Bijaylaxmi Das for providing us with data collection resourses.")

print("="*100)

# Feature extraction parameters
splitting_size = 0
threshold = 1

# hyperparameters
num_epochs = 1000
eta_0 = 1
sigma_0 = 8
t_eta = 500
t_sigma = 650
n = 15

# Find data
Roll_nos = ['17EC32005', '17EC34003', '17EC35023', '17EC35036', '18EC10025', '18EC35010', '18EC35021', '18EC35045', '18EC3AI19']
num_roll_nos = len(Roll_nos)

data_features = []
data_tags = []
unique_features = set([])

# Extract features for each data file
# Roll_nos = ['17EC32005', '17EC34003']
Roll_nos = ['17EC32005', '17EC34003', '17EC35023', '17EC35036', '18EC10025', '18EC35010', '18EC35021', '18EC35045', '18EC3AI19']
# Roll_nos = ['17EC32005', '18EC35021']
num_roll_nos = len(Roll_nos)

data_features = []
data_tags = []
unique_features = set([])


for r in range(num_roll_nos):
    # Make an output prompt
    print("\tExtractiong Features of ", Roll_nos[r]," ...")
    for subdir,dirs,file in os.walk(os.path.join('Continuous Data Rearranged',Roll_nos[r])):
        for directory in dirs:
#             print(subdir,directory)
            features = extract_keylog_features(os.path.join('Continuous Data Rearranged',Roll_nos[r],directory,'Keylog.txt'))
            data_features.append(features)
            data_tags.append(r)
            
            unique_features = unique_features.union(set([ x for (x,y) in features]))
            
print("-"*100)

# Split the files into even sizes
data_features_temp = []
data_tags_temp = []

for r in range(len(Roll_nos)):
    indices = np.where(np.array(data_tags)==r)[0]
    features_temp = []
    for i in indices:
        features_temp = features_temp + data_features[i]
    
    j=0
    j_next=0
    while(j_next < len(features_temp)):
        j_next = min(j + splitting_size, len(features_temp))
        data_features_temp.append(features_temp[j:j_next])
        data_tags_temp.append(r)
        j = j_next
    
data_features = data_features_temp
data_tags = data_tags_temp

# Remove features that have more than threshold value 
unique_features_dict = { list(unique_features)[i]: i for i in range(len(unique_features)) }
m = len(unique_features_dict)
D = len(data_features) # number of data points

x_vectors = np.zeros((D,m))
x_tags = np.zeros(D, dtype=int)
cox_vectors = np.zeros((D,m)) # count of x

# Construct a feature vector
for d in range(D):
    for (a,a_val) in data_features[d]:
        if(a_val > threshold):
            continue
        a_idx = unique_features_dict[a]
        x_vectors[d,a_idx] = x_vectors[d,a_idx] + a_val
        cox_vectors[d,a_idx] = cox_vectors[d,a_idx] + 1
    cnt_locs = np.where(cox_vectors[d,:]>0)
    x_vectors[d,cnt_locs] = x_vectors[d,cnt_locs] / cox_vectors[d,cnt_locs]
    x_tags[d] = data_tags[d]
    
# Find mean, std and normalise
x_means = np.mean(x_vectors, 0)
x_std = np.std(x_vectors, 0)
cox_total = np.sum(cox_vectors, 0)

std0_locs = np.where(x_std==0)
std_locs = np.where(x_std > 0)

x_vectors[:,std0_locs] = (x_vectors[:,std0_locs] - x_means[std0_locs])
x_vectors[:,std_locs] = (x_vectors[:,std_locs] - x_means[std_locs]) / x_std[std_locs]

# Shuffle indices
shuffle_indices = np.random.permutation(D)
x_vectors = x_vectors[shuffle_indices,:]
cox_vectors = cox_vectors[shuffle_indices,:]
x_tags = x_tags[shuffle_indices]

# Output prompt
print("Finished preprocessing data. Training now for ",num_epochs," epochs")

# Initialise SOM
OurSOM = SOM(m,n)

# Train
OurSOM.train(x_vectors, cox_vectors, num_epochs, eta_0, sigma_0, t_eta, t_sigma)

print("="*100)

# Classify
num_clusters = num_roll_nos
best_fit = OurSOM.cluster(x_vectors, cox_vectors, num_roll_nos, num_clusters)

print("Plotting now the unsupervised clusters obtained in the Kohonen Layer. Please enable display for matplotlib" )

# Plot the clusters
plt.clf()
plt.scatter(best_fit[:,0], best_fit[:,1])
plt.xlabel('p')
plt.ylabel('q')
plt.title("Cluster centers in the Kohonen layer formed by unsupervised learning")
plt.savefig('Figures/clustercentres.jpg', bbox_inches='tight', dpi=1000)
plt.show()

print("="*100)

print("Now plotting the winning neurons for each of the training data in the Kohonen layer for pairs of users. Please enable display for matplotlib.")

# Plotting for pairwise classes
tag1s = [0, 4, 0]
tag2s = [3, 5, 1]

for t in range(3):
    plt.clf()
    p_wins,q_wins =  OurSOM.set_of_winning_neurons(x_vectors, cox_vectors)

    tag1 = tag1s[t]
    tag2 = tag2s[t]
    indices1 = np.where(x_tags==tag1)
    indices2 = np.where(x_tags==tag2)
    plt.scatter(p_wins[indices1], q_wins[indices1], marker='+', c='b', s=plt.rcParams['lines.markersize'] ** 2.2)
    plt.scatter(p_wins[indices2], q_wins[indices2], marker='x', c='r', s=plt.rcParams['lines.markersize'] ** 2)
    plt.legend([Roll_nos[tag1], Roll_nos[tag2]])
    plt.xlabel('p')
    plt.ylabel('q')
    plt.title('Plotting the SOM output neuron layer')
    plt.savefig( 'Figures/' +str(t+1) + '.jpg', bbox_inches='tight', dpi=1000)
    plt.show()
    
    
colors = ['#800000', '#9A6324', '#808000', '#469990', '#000075', '#f58231', '#e6194B', '#dcbeff', '#fabed4']
markers = ['+', 'o', 'v',  'x',  '1',  ',',  '*',  '^',  '>']


# Plotting all the classes

print("="*100)

print("Now plotting the winning neurons for each of the training data in the Kohonen layer for all the users. Please enable display for matplotlib.")

p_wins,q_wins =  OurSOM.set_of_winning_neurons(x_vectors, cox_vectors)
c = np.random.randint(1, 9, size=cox_vectors[0].shape)

plt.clf()
for tag in range(9):    
    indices = np.where(x_tags==tag)
    plt.scatter(p_wins[indices], q_wins[indices], marker=markers[tag], c=colors[tag], label = '%s' % Roll_nos[tag], s=plt.rcParams['lines.markersize'] * (2.+0.1*tag))

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('p')
plt.ylabel('q')
plt.title('Plotting the SOM output neuron layer')
plt.savefig('Figures/Clustered.jpg', bbox_inches='tight', dpi=1000)
plt.show()


accuracies = OurSOM.accuracy(x_vectors, cox_vectors, x_tags, best_fit)

print("="*100)
print("Printing the accurcaies on training set for each cluster and roll number")

accuracy_chars = np.char.mod('%d', accuracies)
print("\tClusters  ", end="")
for c in range(num_clusters):
    print( "|  c" + str(c) + " ",end="")
print("")
for r in range(num_roll_nos):
    print("\t" + "-"*10 + ("+" + "-"*5)*num_clusters)
    print("\t"+Roll_nos[r]+" ", end="")
    for c in range(num_clusters):
        s = "   " + accuracy_chars[r,c]
        print( "| " + s[-3:] + " ",end="")
    print("")

print("="*100)

print("Here is also a heat map of normalised accuracies")
plt.clf()
plt.imshow(accuracies / np.sum(accuracies, axis=1)[:,None], cmap='hot', interpolation='nearest')
plt.savefig('Figures/heatmap.jpg', bbox_inches='tight', dpi=1000)
plt.xlabel("clusters")
plt.yticks(ticks=range(num_roll_nos),labels=Roll_nos)
plt.savefig('Figures/heatmap.jpg', bbox_inches='tight', dpi=1000)
plt.show()

print("="*100)

print("Have a look at the README.md or the report for detailed information.")
print(" "*40 + "THANK YOU")
print("="*100)