from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.externals import joblib
## TODO: Import any additional libraries you need to define a model
#from exceptions import ValueError
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # ----- I DID change SM_CHANNEL_TRAIN' --> SM_CHANNEL_TRAINING' because otherwide I got an error
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])  
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    parser.add_argument('--sklearn_model_name', type=str, default='SVC', choices=['LogisticRegression', 'SVC'])
    # both
    parser.add_argument('--CC', type=float, default=1.0)
    parser.add_argument('--class_weight', type=str, default='balanced')
    # LinearRegression
    parser.add_argument('--penalty', type=str, default='l2')
    parser.add_argument('--solver', type=str, default='lbfgs', 
                        choices=['lbfgs', 'liblinear', 'sag', 'saga', 'newton-cg'])
    # SVC
    parser.add_argument('--kernel', type=str, default='poly', choices=['linear', 'poly', 'rbf'])
    parser.add_argument('--degree', type=int, default=2, choices=[2,3])
    parser.add_argument('--gamma', type=float, default=1.0)
    
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    ## --- Your code here --- ##
    ## TODO: Define a model
    if args.sklearn_model_name=='LogisticRegression':
        model = LogisticRegression(penalty=args.penalty, 
                                   C=args.CC,
                                   class_weight=args.class_weight,
                                   solver=args.solver)
    else:
        model = SVC(class_weight=args.class_weight, 
                    C=args.CC,
                    kernel=args.kernel,
                    degree=args.degree,
                    gamma=args.gamma)
     
    ## TODO: Train the model
    model.fit(train_x, train_y)    
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))