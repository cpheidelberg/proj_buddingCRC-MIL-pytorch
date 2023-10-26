from torch import nn
import torch
import mil
from mil.utils_models import test_loop
from mil.mil import BagModel
import time
def get_model(params, dataset_train = None, train_dl = None, test_dl = None):

    n_classes = params['model']['n_classes']
    n_neurons = params['model']['n_neurons']

    if not dataset_train is None:
        t0 = time.time()
        test = dataset_train[0]
        t1 = time.time()
        print(f"time to load one dataset is {t1-t0}")
        print(f"The input size is {test[0].size()}")

    prepNN = torch.nn.Sequential(
      torch.nn.Linear(dataset_train.data_sizeX, n_neurons),
      torch.nn.ReLU(),
      #torch.nn.Tanh()
    )

    if not dataset_train is None:
        output_prepNN = prepNN(test[0])
        print(f"The output size of prepNN is {output_prepNN.size()}")

    if params['model_mil']['aggregation'] == "mean":
        agg_func = torch.mean
    elif params['model_mil']['aggregation'] == "max":
        def max_along_axis(tensor, dim):
            return torch.max(tensor, dim=dim)[0]
        agg_func = max_along_axis
    elif params['model_mil']['aggregation'] == "attention":
        from mil.aggregation_layer import Aggregation
        agg_func = Aggregation(aggregation_func = torch.mean,
                               linear_nodes=n_neurons,
                               attention_nodes=n_neurons)

    print(f"Used aggregation approach is {params['model_mil']['aggregation']}")

    if not dataset_train is None:
        output_agg_func = agg_func(output_prepNN, dim = 0)
        print(f"The output size of the aggregation function is {output_agg_func.size()}")

    afterNN = torch.nn.Sequential(
      torch.nn.Dropout(0.25),
      torch.nn.Linear(n_neurons, n_classes),
      torch.nn.Softmax(dim = 1)
    )

    if not dataset_train is None:
        #print(output_agg_func)
        #print(output_agg_func.size())
        #print(output_agg_func.T)
        #print(output_agg_func.T.size())
        output_afterNN = afterNN(output_agg_func.T.unsqueeze(dim=0))
        print(f"The output size of afterNN is {output_agg_func.size()}")
        print(f"And the class probability is {output_afterNN}")
        # Define model ,loss function and optimizer

    model = BagModel(prepNN, afterNN, agg_func)

    if not train_dl is None:
        batch_size = 4
        t0 = time.time()
        test_input = next(iter(train_dl))
        t1 = time.time()
        print(f"time to load one batch is {t1-t0}")
        print(f"input data size is {test_input[0].size()}")
        print(f"input bag size is {test_input[0].size()[0]/batch_size}")
        test_output = model((test_input[0], test_input[1])).squeeze()
        print(f"And finally the test output of the mounted model has the size {test_output.size()}")
        print(f"Batch size is {batch_size} and n={n_classes} are used")

    if not test_dl is None:
        # test on validation set
        batch_size = 4
        print("validation accuracy (model pre-training)")
        acc_val_pretraining, _ = test_loop(test_dl, model)

    return model