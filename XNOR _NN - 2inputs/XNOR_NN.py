### 2 - XNOR Neural Network.

import numpy as np;                 #   Numpy for numerical matrix computations.
from scipy.optimize import fmin_bfgs, minimize;


class XNOR_NN:

    def __init__(self, X = None, y = None, num_layers = 2, arr_layer_sizes = [2,1]):
        
        self.var_lambda     = 0;
        self.var_alpha      = 1;
        self.learning_iter  = 20000;

        self.NUM_LAYERS     = num_layers;           ## 3-layers NN: input layer, inter-layer, output-layer.
        self.y              = np.array( y   ).T;    ## y - data output.

        if X:       
            if      type(X) == list:                self.X  = np.array(X);            
            elif    type(X) == np.ndarray:          self.X  = X;
            else:                                   self.X  = np.zeros(  (1,1)   );
              
        else:
            self.X = np.zeros(  (1,1)   );

        m, n                = np.shape(self.X);
        self.X              = np.concatenate(   ( np.ones( (m,1) ), self.X ), 1);

        self.m, self.n      = m, n+1;           #   Initialize num of samples m, and num of features n.
        self.n_inter_layer  = arr_layer_sizes;  #   Initialize the num of nodes of each NN layers.
        self.Thetas         = [None] * (self.NUM_LAYERS - 1);   # Array to hold Thetas for each layer.

        self.init_thetas();                     #   Initialize random thetas (weights).

        self.z      =   [];
        self.a      =   [];

        return;
    
    def flatten_Thetas(self):
        flattened_Thetas = np.array([]);
        for theta in self.Thetas:
            flattened_Thetas = np.concatenate(  (flattened_Thetas, theta.flatten()), 0);
        
        return flattened_Thetas;

    def unflatten_Thetas(self, flatten_Thetas):
        new_Thetas  =   [];
        i_start     =   0;

        for i_layer in range(   self.NUM_LAYERS-1):
            rows, cols          =   np.shape(self.Thetas[i_layer]);
            size                =   rows * cols;
            ft                  =   flatten_Thetas[ i_start : i_start+size];            
            theta               =   ft.reshape( rows, cols);

            new_Thetas.append(  theta  );
            i_start += size;
        
        self.Thetas =   new_Thetas;
        return new_Thetas;


    def init_thetas(self):
        ## Initialize Thetas (Weights) for each layers.
        ##  with small random numbers [-1, 1].

        np.random.seed(5);
        EPSILON         = 0.5;     # small constant multiplier.

        for i in range(len(self.Thetas)):

            this_size, next_size    = self.n_inter_layer[i:i+2];    # Theta (weight) matrix sizes for each NN layer.
               
            theta_layer     =   np.random.random(   ( next_size, this_size + 1));
            theta_layer     =   0 + EPSILON * ( (2 * theta_layer) - 1);  

            self.Thetas[i]   = theta_layer;
        
        return;
        

    def train_NN_with_fmin( self):

        #def decorated_cost(flatenned_thetas):
        #    return self.Cost_function_reg(  flatenned_thetas);
        
        init_flattened_thetas   =   self.flatten_Thetas();

        #fmin_res    = fmin_bfgs(    decorated_cost, init_flattened_thetas, maxiter=400);
        fmin_res    =   fmin_bfgs(  f       = self.Cost_function_reg, \
                                    x0      = init_flattened_thetas, \
                                    fprime  = self.Theta_gradient, \
                                    maxiter = self.learning_iter,
                                    gtol    = (1e-08));
                                    #disp=True, maxiter=400, full_output = True, retall=True);
        
        #f_min_res  =   minimize(    fun     = decorated_cost, \
        #                            x0      = init_flattened_thetas, \
        #                            args    = (self.X, self.y), \
        #                            method  = 'TNC', \
        #                            jac     = Gradient
        #                        );

        #print(fmin_res);
        return;

    def Theta_gradient( self, flattened_thetas):

        a           =   [None] * self.NUM_LAYERS        ##  activation layers.
        z           =   [None] * self.NUM_LAYERS        ##  z = a * theta'.

        a[0]        =   self.X;
        z[0]        =   None;
        m           =   self.m;

        self.Thetas =   self.unflatten_Thetas(  flattened_thetas);

        for i_layer in range(   self.NUM_LAYERS-1):

            theta           =   self.Thetas[ i_layer];
            i_next          =   i_layer + 1;

            z[i_next]       =   np.dot( a[i_layer], theta.T);
            a[i_next]       =   np.concatenate(   ( np.ones( (m, 1)) , self.sigmoid(z[i_next]) ), 1);


        a[-1]               =   a[-1][:,1:];
        self.a, self.z      =   a, z;

        ## Back-propagation.    
        d                   =   self.a[-1] - self.y;

        new_Thetas_grad = np.array([]);

        for i_layer in range(   self.NUM_LAYERS-1):

            theta                   =   self.Thetas[-i_layer-1];
            theta_row, theta_col    =   np.shape(   theta   ); 

            theta_grad      =   (1/m)   *  np.dot(  d.T, a[-i_layer-2]   ); 

            theta_reg_term  =   (self.var_lambda/m)  *   \
                                np.concatenate( (np.zeros( (theta_row, 1)), theta[:,1:]), 1);

            theta_grad      +=  theta_reg_term;     ## adding the regulation terms.
            theta_grad      *=  self.var_alpha;     ## Use constant multiplier for faster descent.

            #self.Thetas[-i_layer-1] -= theta_grad;
            new_Thetas_grad = np.concatenate( (theta_grad.flatten(), new_Thetas_grad), 0); 

            

            if  i_layer < self.NUM_LAYERS-2:

                g_grad  =   self.sigmoid_gradient(  z[-i_layer-2]);
                d       =   np.dot(  d, theta[:,1:])    *   g_grad;

        return new_Thetas_grad.flatten();

    def Cost_function_reg(self, flattened_thetas):

        a           =   [None] * self.NUM_LAYERS        ##  activation layers.
        z           =   [None] * self.NUM_LAYERS        ##  z = a * theta'.

        a[0]        =   self.X;
        z[0]        =   None;
        m           =   self.m;

        self.Thetas =   self.unflatten_Thetas(  flattened_thetas);

        for i_layer in range(   self.NUM_LAYERS-1):

            theta           =   self.Thetas[ i_layer];
            i_next          =   i_layer + 1;

            z[i_next]       =   np.dot( a[i_layer], theta.T);
            a[i_next]       =   np.concatenate(   ( np.ones( (m, 1)) , self.sigmoid(z[i_next]) ), 1);


        a[-1]               =   a[-1][:,1:];
        self.a, self.z      =   a, z;

        J = self.J_cost(    a[-1])  ;       ## Compute cost for each iteration; cost should decrease
                                            ##  per iteration.
        #print("J_cost = ", J);

        return np.array([J]).flatten();

    def train_NN(self):
        
        #vect_zeros  =   np.zeros( (self.m, 1));
        #z           =   vect_zeros;

        a           =   [None] * self.NUM_LAYERS        ##  activation layers.
        z           =   [None] * self.NUM_LAYERS        ##  z = a * theta'.

        a[0]        =   self.X;
        z[0]        =   None;
        m           =   self.m;

        #######################################
        ## Feed-forward. ######################

        for ith in range(self.learning_iter):
            #print(self.Thetas);

            for i_layer in range(   self.NUM_LAYERS-1):

                theta           =   self.Thetas[ i_layer];
                i_next          =   i_layer + 1;

                z[i_next]       =   np.dot( a[i_layer], theta.T);
                a[i_next]       =   np.concatenate(   ( np.ones( (m, 1)) , self.sigmoid(z[i_next]) ), 1);

            a[-1]               =   a[-1][:,1:];
            self.a, self.zip    =   a, z;

            #J = self.J_cost(    a[-1])  ;       ## Compute cost for each iteration; cost should decrease
                                                ##  per iteration.
            #print("J_cost = ", J);
            

            ## Back-propagation.  
            ########################
              
            d                   =   a[-1] - self.y;

            for i_layer in range(   self.NUM_LAYERS-1):

                theta                   =   self.Thetas[-i_layer-1];
                theta_row, theta_col    =   np.shape(   theta   );  

                theta_grad      =   (1/m)   *  np.dot(  d.T, a[-i_layer-2]   ); 

                theta_reg_term  =   (self.var_lambda/m)  *   \
                                    np.concatenate( (np.zeros( (theta_row, 1)), theta[:,1:]), 1);

                theta_grad      +=  theta_reg_term;     ## adding the regulation terms.
                theta_grad      *=  self.var_alpha;     ## Use constant multiplier for faster descent.

                self.Thetas[-i_layer-1] -= theta_grad;

                

                if  i_layer < self.NUM_LAYERS-2:

                    g_grad  =   self.sigmoid_gradient(  z[-i_layer-2]);
                    d       =   np.dot(  d, theta[:,1:])    *   g_grad; 

        return;

    def J_cost(self, H):
        J = 0;
        J_reg_term = 0;

        m, n        =   self.m, self.n;

        for theta in self.Thetas:
            J_reg_term  +=   np.sum(    np.square(theta[:,1:])    );
        
        m_y, n_y = np.shape(    self.y);

        j = np.eye(n_y) *   (   np.dot(  self.y.T,          np.log( H)      )   \
                             +  np.dot(  ( 1.0 - self.y.T ),  np.log( 1.0 - H)  )   \
                            ); 
        
        J = (-1./m) * np.sum(    j   );

        J = J + (   (0.5 * self.var_lambda / m) * J_reg_term    );


        return J;


    def get_Thetas(self):
        return self.Thetas;

    def print_Thetas(self):
        for theta_ith in self.Thetas:
            print(theta_ith, "\n");
        print("\n")
        return;
    
    def sigmoid(self, Z):
        return 1. / (1. + np.exp(-Z));
    
    def sigmoid_gradient(self, Z):
        sg = self.sigmoid(Z);
        return sg * (1. - sg);

    def H_funct(self, X):
        
        h           =   np.concatenate( (np.ones( (1,1)), np.array(X)), 1);
        for theta in self.Thetas:
            h       =   self.sigmoid(    h.dot(theta.T));
            h       =   np.concatenate( (np.ones( (1,1)), h), 1);

        return round(h[0,1]);  
