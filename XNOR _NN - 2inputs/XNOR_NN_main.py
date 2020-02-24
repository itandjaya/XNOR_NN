## XNOR Neural Network Main function for testing.
## XNOR_NN_main.py

import sys;
from XNOR_NN import XNOR_NN;
#import numpy as np;

def main():

    X_inputs        =   [   [0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]  ];

    y_output        =   [[1, 0, 0, 1]];

    xnor_nn        =   XNOR_NN(    X_inputs,       ## input data.
                                    y_output,       ## output data.
                                    3,              ## 3 NN layers: Input, inter-layer, output.
                                    [2,2,1] );      ## num of nodes for each layer.

    TEST1, TEST2 = True, True;

    if TEST1:
        print("\nNeural Network XNOR - using GRADIENT DESCENT ITERATION\n", "#"*30, "\n");
        

        xnor_nn.init_thetas();
        xnor_nn.train_NN();
        print("Final Cost J = ", xnor_nn.J_cost(xnor_nn.a[-1]));

        for test_input in X_inputs:
            test_res   =   xnor_nn.H_funct(    [test_input]    );
            print("XNOR(",test_input, ") => ", test_res);
        
        print("\n");

    if TEST2:
        print("\nNeural Network XNOR - using fmin_bfgs\n", "#"*30, "\n");

        xnor_nn.init_thetas();
        xnor_nn.train_NN_with_fmin();

        for test_input in X_inputs:
            test_res   =   xnor_nn.H_funct(    [test_input]    );
            print("XNOR(",test_input, ") => ", test_res);
        
        print("\n");

    return 0;

if __name__ == '__main__':  main();



