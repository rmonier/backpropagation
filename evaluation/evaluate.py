import os
# import matplotlib.pyplot as plt

#get smallest MAPE error for all datasets
dir_results = os.getcwd() + "/evaluation/results/"
for filename_overview in os.listdir(dir_results):
    if filename_overview.startswith("bp_"):
        dataset = filename_overview.split("-")[1][:-4]
        print(dataset)
        with open(dir_results + filename_overview, 'r') as f_overview:
            print("min error for " + filename_overview + ": ")
            lines = f_overview.readlines()[1:]
            #skip line 0
            min_mape = lines[0].split(",")[-1]
            min_mape_line = -1
            for l in lines:
                Layers,Activation,LearningRate,Momentum,Epochs,MAPEError=l.split(",")
                if (float(min_mape) >float(MAPEError.strip())):
                    min_layers,min_activation,min_learningRate,min_momentum,min_epochs,min_mape=l.split(",")
            print("min values:", min_layers,min_activation,min_learningRate,min_momentum,min_epochs,min_mape)
            
            #plot epoch error for best parameters
            #epochs_bp_A1-synthetic_sigmoid_1000_0.15_0_9-9-1.csv
            #epochs_bp_A1-top10ssigmoid10000.10.79-9-1.csv
            min_layers = min_layers.replace("; ", "-")[1:-1]
            filename_best_paras = os.getcwd() + "/evaluation/results/" + "epochs_bp_A1-" + "_" + dataset + "_" +  min_activation + "_" + min_epochs + "_" + min_learningRate + "_" + min_momentum + "_" + min_layers + ".csv"
            print(filename_best_paras)
            # with open(filename_best_paras, 'r') as f_best:
            