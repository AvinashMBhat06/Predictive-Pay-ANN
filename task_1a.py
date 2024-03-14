'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[GG_2568]
# Author List:		[Avinash M Bhat]
# Filename:			task_1a.py
# Functions:	    [`ideantify_features_and_targets`, `load_as_tensors`,
# 					 `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
# 					 `validation_functions` ]

####################### IMPORT MODULES #######################
import pandas 
import torch 
###################### Additional Imports ####################
'''
You can import any additional modules that you require from 
torch, matplotlib or sklearn. 
You are NOT allowed to import any other libraries. It will 
cause errors while running the executable
'''
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn.init as init
from sklearn.preprocessing import LabelEncoder
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################
##############################################################


def data_preprocessing(task_1a_dataframe):
    '''
    Purpose:
    ---
    This function will be used to load your CSV dataset and preprocess it.
    Preprocessing involves cleaning the dataset by removing unwanted features,
    handling missing values, and encoding textual features into numerical labels.

    Input Arguments:
    ---
    `task_1a_dataframe`: [Dataframe]
                          Pandas dataframe read from the provided dataset

    Returns:
    ---
    `X`: [Dataframe]
         Pandas dataframe with input features (numeric and encoded categorical)
    `y`: [Dataframe]
         Pandas dataframe with target labels (numeric)

    Example call:
    ---
    X, y = data_preprocessing(task_1a_dataframe)
    '''

    #################   ADD YOUR CODE HERE   ##################
        # Drop unnecessary columns (if any)
    encoded_dataframe = task_1a_dataframe.drop(['JoiningYear'], axis=1)

    # Encode textual features using LabelEncoder
    label_encoders = {}  # Dictionary to store label encoders for each categorical column
    categorical_columns = ['Education', 'City', 'PaymentTier', 'Gender', 'EverBenched']

    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        encoded_dataframe[column] = label_encoders[column].fit_transform(encoded_dataframe[column])
    ##########################################################
    return encoded_dataframe
def identify_features_and_targets(encoded_dataframe):
    '''
    Purpose:
    ---
    The purpose of this function is to define the features and
    the required target labels. The function returns a python list
    in which the first item is the selected features and second 
    item is the target label 

    Input Arguments:
    ---
    `encoded_dataframe` : [ Dataframe ]
                        Pandas dataframe that has all the features mapped to 
                        numbers starting from zero
    
    Returns:
    ---
    `features_and_targets` : [ list ]
                            python list in which the first item is the 
                            selected features and second item is the target label

    Example call:
    ---
    features_and_targets = identify_features_and_targets(encoded_dataframe'''
        
        #################	ADD YOUR CODE HERE	##################
    features = encoded_dataframe.columns[:-1]
    features_df = encoded_dataframe[features]
    target_label = encoded_dataframe[encoded_dataframe.columns[-1]]

# Convert the target column to a DataFrame
    target_label_df = pandas.DataFrame(target_label)


    features_and_targets =[features_df,target_label_df]
    ##########################################################
    return features_and_targets



def load_as_tensors(features_and_targets):

    ''' 
    Purpose:
    ---
    This function aims at loading your data (both training and validation)
    as PyTorch tensors. Here you will have to split the dataset for training 
    and validation, and then load them as as tensors. 
    Training of the model requires iterating over the training tensors. 
    Hence the training sensors need to be converted to iterable dataset
    object.
    
    Input Arguments:
    ---
    `features_and targets` : [ list ]
                            python list in which the first item is the 
                            selected features and second item is the target label
    
    Returns:
    ---
    `tensors_and_iterable_training_data` : [ list ]
                                            Items:
                                            [0]: X_train_tensor: Training features loaded into Pytorch array
                                            [1]: X_test_tensor: Feature tensors in validation data
                                            [2]: y_train_tensor: Training labels as Pytorch tensor
                                            [3]: y_test_tensor: Target labels as tensor in validation data
                                            [4]: Iterable dataset object and iterating over it in 
                                                 batches, which are then fed into the model for processing

    Example call:
    ---
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    '''
    #################	ADD YOUR CODE HERE	##################
    features_df, target_label_df = features_and_targets

    # Split data into features and target label
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(features_df, target_label_df, test_size=0.2, random_state=42)

    # Convert features and target label DataFrames to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_df.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_df.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_df.values, dtype=torch.float32)

    # Create TensorDatasets for training and validation sets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoader objects for training and validation set
    batch_size = 500
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    tensors_and_iterable_training_data = [
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader
    ]
    ##########################################################
    return tensors_and_iterable_training_data

class Salary_Predictor(nn.Module):
    def __init__(self):
        super(Salary_Predictor, self).__init__()
        self.fc1 = nn.Linear(7, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

        # Custom weight and bias initialization
        init.xavier_normal_(self.fc1.weight)
        init.constant_(self.fc1.bias, 0.1)
        init.xavier_normal_(self.fc2.weight)
        init.constant_(self.fc2.bias, 0.1)
        init.xavier_normal_(self.fc3.weight)
        init.constant_(self.fc3.bias, 0.1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))  # Apply sigmoid activation
        x = torch.round(x)  # Round the values to 0 or 1
        return x


def model_loss_function():
    '''
    Purpose:
    ---
    To define the loss function for the model. Loss function measures 
    how well the predictions of a model match the actual target values 
    in training data.
    
    Input Arguments:
    ---
    None

    Returns:
    ---
    `loss_function`: This can be a pre-defined loss function in PyTorch
                    or can be user-defined

    Example call:
    ---
    loss_function = model_loss_function()
    '''
    #################	ADD YOUR CODE HERE	##################
    loss_function = nn.BCEWithLogitsLoss()

    ##########################################################
    
    return loss_function

def model_optimizer(model):
    '''
    Purpose:
    ---
    To define the optimizer for the model. Optimizer is responsible 
    for updating the parameters (weights and biases) in a way that 
    minimizes the loss function.
    
    Input Arguments:
    ---
    `model`: An object of the 'Salary_Predictor' class

    Returns:
    ---
    `optimizer`: Pre-defined optimizer from Pytorch

    Example call:
    ---
    optimizer = model_optimizer(model)
    '''
    #################	ADD YOUR CODE HERE	##################
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=0.004)

    ##########################################################

    return optimizer

def model_number_of_epochs():
    '''
    Purpose:
    ---
    To define the number of epochs for training the model

    Input Arguments:
    ---
    None

    Returns:
    ---
    `number_of_epochs`: [integer value]

    Example call:
    ---
    number_of_epochs = model_number_of_epochs()
    '''
    #################	ADD YOUR CODE HERE	##################
    number_of_epochs=100
    ##########################################################

    return number_of_epochs
def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):


    for epoch in range(number_of_epochs):
        model.train()  # Set the model to training mode

        for batch_X, batch_y in tensors_and_iterable_training_data[-1]:
      
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_X)
            # Compute the loss
            loss = loss_function(predictions, batch_y)
            # Backpropagation
            loss.backward()
            optimizer.step()
            
    return model


def validation_function(trained_model, tensors_and_iterable_training_data):
    '''
    Purpose:
    ---
    This function will utilize the trained model to do predictions on the
    validation dataset. This will enable us to understand the accuracy of
    the model.

    Input Arguments:
    ---
    1. `trained_model`: Returned from the training function
    2. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
                                             and iterable dataset object of training tensors
    3. `threshold`: Threshold for converting probabilities to binary predictions (default is 0.5)

    Returns:
    ---
    model_accuracy: Accuracy on the validation dataset
    binary_predictions: Binary predictions (0 or 1) for the validation dataset

    Example call:
    ---
    model_accuracy, binary_predictions = validation_function(trained_model, tensors_and_iterable_training_data)
    '''    
    #################    ADD YOUR CODE HERE    ##################
    # Move data tensors to the determined device
    X_val, y_val = tensors_and_iterable_training_data[1], tensors_and_iterable_training_data[3]

    # Set the model to evaluation mode
    trained_model.eval()

    # Disable gradient computation during validation
    with torch.no_grad():
        # Forward pass on validation data
        predicted_outputs = trained_model(X_val)
        # Convert probabilities to binary predictions using the threshold

    # Cast binary predictions to integers

    # Calculate accuracy
    correct_predictions = (predicted_outputs== y_val).sum().item()
    total_samples = y_val.size(0)
    model_accuracy = correct_predictions / total_samples

    # Set the model back to training mode
    trained_model.train()
    ##########################################################

    return model_accuracy

########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########	
'''
    Purpose:
    ---
    The following is the main function combining all the functions
    mentioned above. Go through this function to understand the flow
    of the script

'''
if __name__ == "__main__":

    # reading the provided dataset csv file using pandas library and 
    # converting it to a pandas Dataframe
    task_1a_dataframe = pandas.read_csv('task_1a_dataset.csv')

    # data preprocessing and obtaining encoded data
    encoded_dataframe = data_preprocessing(task_1a_dataframe)

    # selecting required features and targets
    features_and_targets = identify_features_and_targets(encoded_dataframe)

    # obtaining training and validation data tensors and the iterable
    # training data object
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    
    # model is an instance of the class that defines the architecture of the model
    model = Salary_Predictor()

    # obtaining loss function, optimizer and the number of training epochs
    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()

    # training the model
    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, 
                    loss_function, optimizer)

    # validating and obtaining accuracy
    model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
    print(f"Accuracy on the test set = {model_accuracy}")

    X_train_tensor = tensors_and_iterable_training_data[0]
    x = X_train_tensor[0]

    jitted_model = torch.jit.save(torch.jit.trace(model,x), "task_1a_trained_model.pth")

 