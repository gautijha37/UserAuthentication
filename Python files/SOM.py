import numpy as np

# The Kohonen class
class SOM:
    # Initialise with the inputlayer and kohonen layer dimensions
    def __init__(self, input_dimension, Koho_dimension):
        self.m = input_dimension
        self.n = Koho_dimension
        self.weights = np.random.randn(self.n, self.n, self.m)
    
    # To find the winning neuron for an input
    def winning_neuron(self, x, cox):
        cnt_locs = np.where(cox>0)
        Euclid_dist = np.sum( np.square(x[cnt_locs] - self.weights[:,:,cnt_locs]) , 2)
        num = np.argmin( Euclid_dist )
        p_win, q_win = tuple(np.array([num/self.n, num%self.n],dtype=int))
        return p_win, q_win
    
    # To find nearness of neighbouring neurons to winning neuron
    def neighbourhood_weights(self, p_win, q_win, sigma):
        Lateral_dist_sq = np.zeros((self.n, self.n)) + (np.square(np.arange(self.n) - p_win) 
                                                        + np.square(np.arange(self.n) - q_win)[:,None])
        Topo_neigh = np.exp(-Lateral_dist_sq/(2*np.square(sigma)))
        return Topo_neigh
    
    # Updating weights according to learning rate, Topo parameter
    def update_weights(self, x, cox, eta, Topo_neigh):
        cnt_locs = np.where(cox>0)
        delta_weights = (x[cnt_locs] - self.weights[:,:,cnt_locs])
        self.weights[:,:,cnt_locs] = self.weights[:,:,cnt_locs] + delta_weights
    
    # Function to perform training
    def train(self, x_vectors, cox_vectors, num_epochs, eta_0, sigma_0, t_eta, t_sigma):
        D = x_vectors.shape[0]
        
        # For all epochs
        for t in range(num_epochs):
            eta = eta_0 * np.exp(-t/t_eta)
            sigma = sigma_0 * np.exp(-t/t_sigma)
            
            # For all data
            for d in range(D):
                # COMPETITION
                p_win, q_win = self.winning_neuron(x_vectors[d,:], cox_vectors[d,:])
                # COOPERATION
                Topo_neigh = self.neighbourhood_weights(p_win, q_win, sigma)
                # UPDATE
                self.update_weights(x_vectors[d,:], cox_vectors[d,:], eta, Topo_neigh)
            
            if(t%10==0):
                # OUPUT PROMPT
                print("\tFinished epoch :",t, "\teta = ", round(eta+0.000001,3), "\tsigma =", round(sigma+0.00000001,3))
#                 print(Topo_neigh, eta)

    
    # outputs set of winning neurons for given feature set
    def set_of_winning_neurons(self, x_vectors, cox_vectors):
        D = x_vectors.shape[0]
        p_wins = np.zeros(D)
        q_wins = np.zeros(D)
        for d in range(D):
            p_wins[d], q_wins[d] = self.winning_neuron(x_vectors[d,:], cox_vectors[d,:])
        
        return p_wins, q_wins
    
    # A fxn to compute Euclidean distance
    def eucd_dist(self, p1, p2):
        dist = np.sqrt(np.sum(np.square(p1-p2)))
        return dist

    # A function to initialise with random neurons
    def init_mean(self, num_points, data):
        ind = np.random.choice(len(data),num_points, replace=False)
        centers = np.zeros((num_points, data.shape[1]))
        for i in range(num_points):
            centers[i] = data[int(ind[i])]
        return centers

    # A fxn to classify a given data according to our means
    def classify(self, c_means, data):
        class_val = np.zeros((len(data)), dtype=int)
        for i in range(len(data)):
            min_dist = 1000000
            ind = 0
            for j in range(len(c_means)):
                dist = self.eucd_dist(c_means[j], data[i])
                if(dist<min_dist):
                    min_dist = dist
                    ind = j
            class_val[i] = ind
        return class_val

    # A fxn to update cluster centers
    def update(self, c_means, new_class, data):
        new_means = np.zeros((c_means.shape))
        means_count = np.zeros((c_means.shape[0]))

        for i in range(len(data)):
            new_means[new_class[i]] = new_means[new_class[i]] + data[i]
            means_count[new_class[i]] += 1

        for j in range(len(means_count)):
            new_means[j] /= (float(means_count[j])+0.00001)

        return new_means

    # A fxn to perform k means clustering for some number of iterations
    def cluster(self, x_vectors, cox_vectors, num_clusters=9, num_iters=10):
        D = x_vectors.shape[0]
        winning_neurons = np.zeros((D,2))
        cluster_vals = np.zeros(D, dtype=int)
        for d in range(D):
            winning_neurons[d,:] = self.winning_neuron(x_vectors[d,:],cox_vectors[d,:])
            
        new_means = self.init_mean(num_clusters, winning_neurons)

        for i in range(num_iters):

            labels = self.classify(new_means, winning_neurons)
            new_means = self.update(new_means, labels, winning_neurons)

        return new_means
    
    # Finds the accuracy for a given dataset
    def accuracy(self, x_vectors, cox_vectors, x_tags, best_fit_clstr_ctrs):
        D = x_vectors.shape[0]
        unique_tags = np.unique(x_tags)
        num_tags = unique_tags.shape[0]
        num_clusters = best_fit_clstr_ctrs.shape[0]
        accuracy_count = np.zeros((num_tags, num_clusters))
        for d in range(D):
            winning_neuron = self.winning_neuron(x_vectors[d,:],
                                                 cox_vectors[d,:])
            clstr_idx = np.argmin(np.sum(np.square(winning_neuron 
                                                   -best_fit_clstr_ctrs),1))
            accuracy_count[x_tags[d],clstr_idx] = accuracy_count[x_tags[d],
                                                                 clstr_idx] + 1
            
        return accuracy_count
            
            