#################
#### Imports ####
#################
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import ctypes
from ctypes import POINTER, c_int, c_float
import functools
import io
import hashlib
import json
import matplotlib.pyplot as plt

###################
#### Constants ####
###################

FENS_TO_KEEP = 30000 if __name__ == "__main__" else 7153281
# Max possible value for FENS_TO_KEEP is 7153281 for Noah.epd
# Higher values will give an error because we don't have enough fens
# Set to something like 100000 for debugging purposes

VALIDATION_PERCENT = 10 if __name__ == "__main__" else 0 # Don't mess with the training runs yet, I guess
TRAINING_FENS = math.ceil(FENS_TO_KEEP * (1 - VALIDATION_PERCENT / 100))
DATASET_TYPES = ["train", "val", "all"]

##################################
#### Functions defined in C++ ####
##################################

# Step 1: Load the library
lib_path = "libamethyst_tuner.dylib"
libtuner = ctypes.cdll.LoadLibrary(lib_path)

# Step 2: Set the argtypes of all functions from the library
libtuner.getResults.argtypes = [POINTER(c_float), c_int]
libtuner.getPhases.argtypes = [POINTER(c_int), c_int]
libtuner.getKingSquares.argtypes = [c_int, POINTER(c_int), c_int]
libtuner.getMobility.argtypes = [c_int, POINTER(c_int), c_int]
libtuner.getPSTs.argtypes = [c_int, c_int, c_int, POINTER(c_int), c_int]

@functools.cache
def get_results(dataset_type : str = "all"):
    # I could optimize this not to calculate 3 different times for "train", "val", and "all", but it's so fast it doesn't matter
    results = (c_float * FENS_TO_KEEP)()
    libtuner.getResults(results,
                       c_int(FENS_TO_KEEP))
    results = np.ctypeslib.as_array(results, shape = (FENS_TO_KEEP,))
    results = torch.tensor(results)
    if dataset_type == "train":
        return results[:TRAINING_FENS]
    elif dataset_type == "val":
        return results[TRAINING_FENS:]
    elif dataset_type == "all":
        return results
    else:
        raise ValueError(f"Unrecognized value: {dataset_type=}")

@functools.cache
def get_phases(dataset_type : str = "all"):
    # I could optimize this not to calculate 3 different times for "train", "val", and "all", but it's so fast it doesn't matter
    phases = (c_int * FENS_TO_KEEP)()
    libtuner.getPhases(phases,
                       c_int(FENS_TO_KEEP))
    phases = np.ctypeslib.as_array(phases, shape = (FENS_TO_KEEP,))
    phases = torch.tensor(phases)
    phases = torch.stack([phases, 24 - phases])
    phases = torch.transpose(phases, 0, 1).float()
    if dataset_type == "train":
        return phases[:TRAINING_FENS,:]
    elif dataset_type == "val":
        return phases[TRAINING_FENS:,:]
    elif dataset_type == "all":
        return phases
    else:
        raise ValueError(f"Unrecognized value: {dataset_type=}")

def get_king_squares(side):
    kingSquares = (c_int * FENS_TO_KEEP)()
    libtuner.getKingSquares(c_int(side), kingSquares, c_int(FENS_TO_KEEP))
    kingSquares = np.ctypeslib.as_array(kingSquares, shape=(FENS_TO_KEEP,))
    return kingSquares.reshape(-1,1)

def get_mobility(side):
    mobility = (c_int * (6 * FENS_TO_KEEP))()
    libtuner.getMobility(c_int(side), mobility, c_int(FENS_TO_KEEP))
    mobility = np.ctypeslib.as_array(mobility, shape=(6 * FENS_TO_KEEP,))
    return mobility.reshape(-1,6)
    # We will convert to torch tensor and to float when we stack all the tensors together

def get_psts(side, piece):
    maxPieces = [8,2,2,2,2,1][piece]
    psts = (c_int * (maxPieces * FENS_TO_KEEP))()
    libtuner.getPSTs(c_int(side),      # side
                     c_int(piece),     # piece
                     c_int(maxPieces), # maxPieces
                     psts,             # arr
                     c_int(FENS_TO_KEEP))      # length
    
    psts = np.ctypeslib.as_array(psts, shape = (maxPieces * FENS_TO_KEEP,))
    psts = psts.reshape(-1,maxPieces)
    psts.sort()
    return torch.tensor(psts)

############################
#### King bucketed PSTs ####
############################

BLACK_KING_BUCKETS = [0, 0, 0, 0, 1, 1, 1, 1,
                      0, 0, 0, 0, 1, 1, 1, 1,
                      0, 0, 0, 0, 1, 1, 1, 1,
                      0, 0, 0, 0, 1, 1, 1, 1,
                      0, 0, 0, 0, 1, 1, 1, 1,
                      0, 0, 0, 0, 1, 1, 1, 1,
                      0, 0, 0, 0, 1, 1, 1, 1,
                      0, 0, 0, 0, 1, 1, 1, 1]

NUM_KING_BUCKETS = 2

def get_king_bucketed_psts(side, piece):
    psts = get_psts(side, piece)
    king_squares = get_king_squares(side).reshape(-1)
    cap = torch.tensor(64 * NUM_KING_BUCKETS)
    
    king_buckets = torch.tensor([BLACK_KING_BUCKETS[ksq ^ 56] for ksq in king_squares]).reshape(-1, 1)
    
    psts = torch.where(psts == 64, torch.tensor(10000), psts)
    psts += 64 * king_buckets
    psts = torch.minimum(psts, cap)
    return psts

@functools.cache
def get_stacked_king_bucketed_psts(piece):
    return torch.stack([get_king_bucketed_psts(side,piece) for side in (0,1)])

@functools.cache
def get_embeddings(dataset_type : str = "all"):
    king_bucketed_psts = []
    for piece in range(6):
        these_psts = get_stacked_king_bucketed_psts(piece)
        if dataset_type == "train":
            king_bucketed_psts.append(these_psts[:,:TRAINING_FENS,:])
        elif dataset_type == "val":
            king_bucketed_psts.append(these_psts[:,TRAINING_FENS:,:])
        elif dataset_type == "all":
            king_bucketed_psts.append(these_psts)
        else:
            raise ValueError(f"Unrecognized value: {dataset_type=}")
    return king_bucketed_psts

###############################################
#### Making the continuous features tensor ####
###############################################

# This could probably be made into a dataclass
class ContinuousFeature(object):
    def __init__(
        self,
        name,
        dims,
        func
    ):
        self.name = name
        self.dims = dims
        self.func = func

    def get_diff(self):
        return self.func(0) - self.func(1)

CONTINUOUS_FUNCTIONS = [
    ContinuousFeature(
        "mobility",
        6,
        get_mobility
    )
]


def get_continuous_tensor_(features: list[ContinuousFeature]):
    continuous_tensors = []
    for feature in features:
        tensor = feature.get_diff()
        tensor = torch.tensor(tensor)
        continuous_tensors.append(tensor)
    result = torch.hstack(continuous_tensors)
    result = result.float()
    return result

@functools.cache
def get_continuous_tensor(dataset_type : str = "all"):
    if dataset_type == "all":
        return get_continuous_tensor_(CONTINUOUS_FUNCTIONS)
    elif dataset_type == "train":
        return get_continuous_tensor(dataset_type = "all")[:TRAINING_FENS,:]
    elif dataset_type == "val":
        return get_continuous_tensor(dataset_type = "all")[TRAINING_FENS:,:]
    else:
        raise ValueError(f"Unrecognized value: {dataset_type=}")

###################################
#### Initialization of tensors ####
###################################

def initialize_tensors():
    t0 = time.time()
    for dataset_type in DATASET_TYPES:
        get_embeddings(dataset_type)
    t1 = time.time()
    print(f"Made embeddings in {t1-t0:.2f} seconds")

    t0 = time.time()
    for dataset_type in DATASET_TYPES:
        get_continuous_tensor(dataset_type)
    t1 = time.time()
    print(f"Made continuous tensor in {t1-t0:.2f} seconds")


    t0 = time.time()
    for dataset_type in DATASET_TYPES:
        get_results(dataset_type)
    t1 = time.time()
    print(f"Made results tensor in {t1-t0:.2f} seconds")

    t0 = time.time()
    for dataset_type in DATASET_TYPES:
        get_phases(dataset_type)
    t1 = time.time()
    print(f"Made phase tensor in {t1-t0:.2f} seconds")

#########################################
#### Tuner class, defining the model ####
#########################################

EMBEDDING_DIMS = [128, 128, 128, 128, 128, 128]
# EMBEDDING_DIMS is how high the numbers in each embedding go
# And how big to make the nn.Embedding corresponding to it
# So if EMBEDDING_DIMS[0] is 64, a Tuner will have its first embedding layer be a nn.Embedding(65, 2, padding_idx = 64)

SCALING_TERM = 28 * 400 / 2.1 
# Yes that should be a 24 instead of 28 and that's on my to do list to change
# Unfortunately it's not as simple as "just change it"
# The "28" typo effectively uses a different scaling term than 400
# This changes the weights of everything

class Tuner(nn.Module):
    def __init__(self):
        super(Tuner, self).__init__()
        self.continuous_layer = nn.Linear(get_continuous_tensor().shape[1], 2, bias = False)
        with torch.no_grad():
            self.continuous_layer.weight.zero_()

        self.embedding_layers = nn.ModuleList()
        for dim in EMBEDDING_DIMS:
            layer = nn.Embedding(dim + 1, 2, padding_idx = dim)
            with torch.no_grad():
                layer.weight.zero_()
            self.embedding_layers.append(layer)
                
    def get_logits(
        self, 
        layer, 
        phases, 
        embedding_tensor,
    ):
        embeddings = layer(embedding_tensor)
        summed = torch.sum(embeddings, axis = 2)
        summed2 = summed[0] - summed[1]
        return torch.sum(summed2 * phases, axis = 1)
        
        
    def forward(self, phases, continuous_tensor, embedding_tensors):
        assert len(embedding_tensors) == len(self.embedding_layers)
        logits = sum(self.get_logits(layer, phases, embedding_tensors[i]) for i, layer in enumerate(self.embedding_layers))
        logits += torch.sum(phases * self.continuous_layer(continuous_tensor), axis = 1)
        return torch.sigmoid(logits / SCALING_TERM)
    
    def forward_as_in_engine(self, phases, continuous_tensor, embedding_tensors):
        # This is the same as forward, but it divides by 24 instead of SCALING_TERM and it doesn't take the sigmoid
        logits = sum(self.get_logits(layer, phases, embedding_tensors[i]) for i, layer in enumerate(self.embedding_layers))
        logits += torch.sum(phases * self.continuous_layer(continuous_tensor), axis = 1)
        # We don't need to worry about setting these to ints. The other thing will do it for us.
        return logits / 24
    
    def round_weights(self):
        with torch.no_grad():
            self.continuous_layer.weight.round_()
            for layer in self.embedding_layers:
                layer.weight.round_()

    def get_weird_l2_norm(self): # We only take the L2 norm of embeddings
        return sum(torch.var(p[:-1,:], axis = 0, correction = 0).sum() for p in self.parameters() if p.shape[1] == 2) / (SCALING_TERM / 24) ** 2
        # Instead of taking the linear product of 1-hot tensors and stuff and then a sigmoid,
        # We take the linear product, then multiply by 24 implicity (phases), then divide by SCALING_TERM, then take the sigmoid.
        # Therefore weights need to be SCALING_TERM / 24 times bigger than they would normally be.
        # Therefore the L2 norm is (SCALING_TERM / 24)^2 times bigger than it should be.
        # So we divide it by (SCALING_TERM / 24)^2

        # We only do L2 regularization on the PSTs, not on any of the continuous features
        # The theory is, I noticed weights for some pieces differing by 100 cp between squares right next to each other.
        # That is almost certainly due to noise in the training data
        # So, we want the PST values to be close to each other
        # For this reason, we regularize the variance of the piece square tables
        # Things we do NOT regularize:
        # Continuous features: These are much more signal than noise
        # Material weights: This is why we have a penalty for variance, not sum of squares of weights. If a queen still has the same "best squares" but gains value, this should not be regularized.
        # Difference between midgame and endgame weights: This is likely signal, not noise. This is why we track variance differently for each phase.
        # TODO: explain this better


    def get_hash(self):
        # NOTE: This gives a different result depending on if you round weights first or not!
        # Method adapted from https://stackoverflow.com/questions/63880081/how-to-convert-a-torch-tensor-into-a-byte-string
        buff = io.BytesIO()
        torch.save(self.continuous_layer.weight, buff)
        for emb in self.embedding_layers:
            torch.save(emb.weight, buff)
        buff.seek(0)
        b = buff.read() # This is a bytes object
        return hashlib.sha256(b).hexdigest() # This is a 64 character long hex string
        # sha256 is slightly overkill, but it doesn't really matter. This function runs in no time anyway.

##########################
#### Training process ####
##########################

def get_detached_params(model):
    return [param.detach().clone() for param in model.parameters()]

def get_max_delta(params1, params2):
    overall_max_delta = 0
    for param1, param2 in zip(params1, params2):
        deltas = torch.abs(param2 - param1)
        max_delta = torch.max(deltas).item()
        overall_max_delta = max(max_delta, overall_max_delta)
        
    return overall_max_delta

def get_model(
    model = None, 
    optimizer = "lbfgs", 
    max_epochs = 3200, 
    verbose = True,
    run_data : dict = {},
    l2_lambda = 0.01,
    base_lr = 1e-3
):
    # 1. Random stuff we need
    t0 = time.time()
    if model is None:
        model = Tuner()
    optimizer_str = optimizer

    run_data["hash"] = []
    run_data["loss"] = []
    run_data["bce_loss"] = []
    run_data["max_delta"] = []
    if VALIDATION_PERCENT > 0:
        run_data["val_loss"] = []

    if optimizer == "sgd":
        lr = base_lr * (SCALING_TERM / 24) ** 2
    elif optimizer in ("adam", "adamw"):
        lr = base_lr * (SCALING_TERM / 24)
    # as established before, weights are SCALING_TERM / 24 times bigger than they "should" be
    # Gradients wrt weights are SCALING_TERM / 24 times smaller than they "should" be e.g. changing a pawn weight from 1 to 2 changes the loss a lot, not so much if you change the pawn weight from 100 to 101
    # Also the gradients need to be SCALING_TERM / 24 times bigger than normal to move weights the "same" amount
    # For sgd:
    # These combine, so we need lr to be (SCALING_TERM / 24) ** 2 times bigger
    # For adam:
    # Gradients wrt weights are SCALING_TERM / 24 times smaller than they "should" be, yes
    # but step sizes are the same. Step sizes are just mt / sqrt(vt)
    # So making gradients SCALING_TERM / 24 times smaller doesn't actually make the steps any bigger or smaller
    # So we only need to multiply gradients by SCALING_TERM / 24 once
    
    # 2. Optimizer, loss function, closure for LBFGS
    loss_function = nn.BCELoss()
    if optimizer_str == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters())
    elif optimizer_str in ("adam", "adamw"):
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    elif optimizer_str == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    else:
        assert False # Optimizer must be "lbfgs" or "adam" or "sgd"
    
    def closure():
        # This is the closure required for LBFGS
        optimizer.zero_grad()
        y_pred = model(get_phases("train"), get_continuous_tensor("train"), get_embeddings("train"))
        loss = loss_function(y_pred, get_results("train"))
        l2_norm = model.get_weird_l2_norm()
        loss += l2_lambda * l2_norm
        loss.backward()
        return loss
    
    # 3. Stopping criteria
    STOPPING_DELTA = 0.5
    previous_params = get_detached_params(model)
    current_params = get_detached_params(model)
    max_delta = torch.inf
    
    # 4. Start the training loop
    for epoch in range(1, max_epochs + 1):
        # 4A. Forward pass, loss, backpropagation
        y_pred = model(get_phases("train"), get_continuous_tensor("train"), get_embeddings("train"))
        loss = loss_function(y_pred, get_results("train"))
        bce_loss = loss.item()

        # L2 regularization
        if optimizer != "adamw":
            l2_norm = model.get_weird_l2_norm()
            loss += l2_lambda * l2_norm
            loss.backward()

        if optimizer_str == "lbfgs":
            optimizer.step(closure)
        elif optimizer == "adamw":
            optimizer.step()
            # Weight decay
            # I'm pretty sure we don't scale this by any power of SCALING_TERM / 24
            # We decay the weights by, say 1%. And we still do that whether the weights are 0.01 or 100
            with torch.no_grad():
                for p in model.parameters():
                    if p.shape[1] == 2: # aka if p is an embedding layer (hacky solution but I couldn't figure out anything else)
                        p[:-1,:] -= (p[:-1,:] - p[:-1,:].mean(axis = 0)) * l2_lambda
                        # decay all PSTs to the mean of that PST, indexed by piece type and phase
                        # e.g. the weight for a midgame white knight on f3 / black knight on f6 decays towards the mean weight of a knight in the midgame
                        # the p[:-1,:] part is so that we don't decay the padding index. That needs to stay at 0
                        # I think we should use 2 * l2_lambda because the derivative of w^2 is 2w, but I think that's not standard convention.

        else:
            optimizer.step()
        optimizer.zero_grad()

        # 4B. Printouts
        t1 = time.time()
        check_max_delta = optimizer_str == "lbfgs" or (epoch % 10 == 9)
        if check_max_delta:
            current_params = get_detached_params(model)
            max_delta = get_max_delta(previous_params, current_params)
            previous_params = current_params

        if VALIDATION_PERCENT > 0:
            with torch.no_grad():
                y_pred_val = model(get_phases("val"), get_continuous_tensor("val"), get_embeddings("val"))
                val_loss = loss_function(y_pred_val, get_results("val")).item()

        message = f"epoch {epoch}; t={int(t1-t0)}; bce_loss={bce_loss:.4f}; loss={loss.item():.4f}"
        if VALIDATION_PERCENT > 0:
            message += f"; {val_loss=:.4f}"
        if check_max_delta:
            message += f"; max(∆w) = {max_delta:.4f}"
        if verbose:
            print(message)
            
        # 4C. Add data
        run_data["hash"].append(model.get_hash())
        run_data["bce_loss"].append(bce_loss)
        run_data["loss"].append(loss.item())
        run_data["max_delta"].append(max_delta)
        if VALIDATION_PERCENT > 0:
            run_data["val_loss"].append(val_loss)

        # 4D. Stopping criteria
        # If none of the params have changed more than 0.5 since last time we checked, we break out of the loop.
        if max_delta < STOPPING_DELTA:
            break

    # 5. Return the model
    model.round_weights()
    return model

########################################################
#### Printing weights, saving evals, graphing loss  ####
########################################################

eval_t_name = "packed_eval_t"
def print_king_bucketed_psts(model: Tuner):
    params = model.embedding_layers
    print(f"    constexpr std::array<std::array<std::array<{eval_t_name}, 64>, 6>, {NUM_KING_BUCKETS}> psts = " + "{{")
    for bucket in range(NUM_KING_BUCKETS):
        idx = bucket * 64
        print(" " * 10 + "{{")
        for piece_type in range(6):
            print(" " * 19 + "{")

            # Determine how much whitespace we need
            # TODO: Change this so it only considers this bucket
            l1 = len(str(torch.min(params[piece_type].weight.data.int()).item()))
            l2 = len(str(torch.max(params[piece_type].weight.data.int()).item()))
            l = max(l1, l2)

            for i, thing in enumerate(params[piece_type].weight.data.int()[idx:idx+64,:]):
                print(f"{" " * 23 * (1 - i % 8)}S({thing[0]:{l}},{thing[1]:{l}}), ", end = "\n" * (i % 8 - 6))
            print(" " * 19 + "{")
        print(" " * 10 + "}},")
    print(" " * 8 + "}}; // hash=" + f"{model.get_hash()}")

def print_continuous_weights(model: Tuner):
    # Step 1: Continuous weights
    weights_tensor = model.continuous_layer.weight
    index = 0
    for feature in CONTINUOUS_FUNCTIONS:
        feature_name = feature.name
        dims = feature.dims
        feature_weights = torch.round(weights_tensor[:, index:index+dims])
        str_feature_weights = [f"S{int(feature_weights[0, i]), int(feature_weights[1, i])}" for i in range(feature_weights.shape[1])]
        strw2 = "{" + ", ".join(str_feature_weights) + "}"
        line = f"constexpr {eval_t_name} {feature_name}[{dims}] = {strw2};"
        print(line)
        index += dims
    print(f"// hash={model.get_hash()}")

# IIRC this function is used to make sure the engine and tuner evals match exactly
def save_eval_to_file(model, filename):
    evals = model.forward_as_in_engine(get_phases(), get_continuous_tensor(), get_embeddings())
    with open(filename, "w") as file:
        for eval_ in evals:
            file.write(str(int(eval_.item())))
            file.write('\n')
            
    print(f"len(evals) is {len(evals)}")
    file.close()

def compare_loss_graph(
    losses : list,
    loss_labels : list,
    title : str = "Loss",
    start_epoch = 1, # If this is, say, 200, we will start our graph at epoch 200
    show : bool = True,
    path : str = "" # If path is not the empty string, will save the file to that path
):
    # Code adapted from https://www.geeksforgeeks.org/machine-learning/l1l2-regularization-in-pytorch/
    plt.figure(figsize=(10, 5))
    for loss, label in zip(losses, loss_labels):
        plt.plot(range(start_epoch, len(loss) + 1), loss[start_epoch - 1:], label = label)

    # Source: https://stackoverflow.com/questions/26337493/combine-multiple-line-labels-in-legend
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='best')
    # End the part that's from stack overflow

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    if path:
        plt.savefig(path)
    if show:
        plt.show()

def run_data_loss_graph(run_data: dict, title = "Loss", **kwargs):
    compare_loss_graph(
        [run_data["bce_loss"], run_data["loss"]],
        ["bce_loss", "loss"],
        **kwargs
    )

###############
#### Tests ####
###############

def sha256torch(x):
    # See Tuner.get_hash
    buff = io.BytesIO()
    torch.save(x, buff)
    buff.seek(0)
    b = buff.read() # This is a bytes object
    return hashlib.sha256(b).hexdigest() # This is a 64 character long hex string

def get_ctypes_hashes():
    hashes = [
        sha256torch(x) for x in get_embeddings("train")
    ] + \
    [
        sha256torch(x) for x in get_embeddings("val")
    ] + \
    [
        sha256torch(x) for x in get_embeddings("all")
    ] + \
    [
        sha256torch(func(dataset_type)) for dataset_type in DATASET_TYPES for func in [get_results, get_phases, get_embeddings]
    ]
    hashes_dict = {"hashes":hashes}
    return hashes_dict

def overwrite_ctypes_hashes():
    hashes = get_ctypes_hashes()
    with open("ctypes_hashes.txt", "w") as file:
        json.dump(hashes, file, indent=2)

def check_ctypes_hashes():
    with open("ctypes_hashes.txt", "r") as file:
        loaded_hashes = json.load(file)
    calculated_hashes = get_ctypes_hashes()
    
    if loaded_hashes == calculated_hashes:
        print("ctypes hashes match")
    else:
        print("ctypes hashes DO NOT match")
        print("--- Hashes from file ---")
        print(loaded_hashes)
        print("--- Hashes from calculation ---")
        print(calculated_hashes)

def get_minirun_hashes():
    run_data = {}
    model1 = get_model(
        max_epochs=16, 
        optimizer = "adam", 
        verbose=False,
        run_data = run_data
    )
    return run_data

def overwrite_minirun_hashes():
    hashes = get_minirun_hashes()
    with open("minirun_hashes.txt", "w") as file:
        json.dump(hashes, file, indent=2)

def check_minirun_hashes():
    with open("minirun_hashes.txt", "r") as file:
        loaded_hashes = json.load(file)
    calculated_hashes = get_minirun_hashes()
    
    if loaded_hashes == calculated_hashes:
        print("minirun hashes match")
    else:
        print("minirun hashes DO NOT match")
        print("--- Hashes from file ---")
        print(loaded_hashes)
        print("--- Hashes from calculation ---")
        print(calculated_hashes)

def check_all_hashes():
    check_ctypes_hashes()
    check_minirun_hashes()

def overwrite_all_hashes():
    overwrite_ctypes_hashes()
    overwrite_minirun_hashes()

def print_tensor_shapes():
    print(f"Embeddings shapes: {[[f"{dataset_type}: {t.shape}" for t in get_embeddings(dataset_type)] for dataset_type in DATASET_TYPES]}")
    print(f"Continuous tensor shape: \n{ {dataset_type : get_continuous_tensor(dataset_type).shape for dataset_type in DATASET_TYPES} }")
    print(f"Results tensor shape: \n{ {dataset_type : get_results(dataset_type).shape for dataset_type in DATASET_TYPES} }")
    print(f"Phases tensor shape: \n{ {dataset_type : get_phases(dataset_type).shape for dataset_type in DATASET_TYPES} }")

def print_tensor_dtypes():
    print(f"Embeddings dtypes: {[t.dtype for t in get_embeddings()]}")
    print(f"Continuous tensor dtype: {get_continuous_tensor().dtype}")
    print(f"Results tensor dtype: {get_results().dtype}")
    print(f"Phases tensor dtype: {get_phases().dtype}")

def print_null_model():
    print_continuous_weights(Tuner())
    print_king_bucketed_psts(Tuner())

def print_single_epoch_model():
    model = get_model(
        optimizer = "lbfgs",
        max_epochs=1, 
        verbose=False
    )
    print_continuous_weights(model)
    print_king_bucketed_psts(model)

def print_huge_l2_run():
    run_data = {}
    try:
        tuner = get_model(
            optimizer = "sgd",
            l2_lambda = 1e2,
            run_data = run_data,
            max_epochs = 32
        )
        return tuner, run_data
    except RuntimeError as e:
        print(f"we got a RuntimeError: {e}")

def print_weight_decay():
    model = get_model(optimizer="adam", base_lr = 1, max_epochs=10)
    print_king_bucketed_psts(model)
    for p in model.parameters():
        if p.shape[1] == 2:
            bucketed_means = p[:-1,:].mean(axis = 0)
            print(bucketed_means)
            with torch.no_grad():
                print(p)
                p[:-1,:] -= (p[:-1,:] - bucketed_means) * 0.01
                print(p)
                return
                

    # print_king_bucketed_psts(model)

def ctypes_shapes_test() -> bool:
    passed = True

    # 1. King bucketed PSTs
    for i, tensor in enumerate(get_embeddings()):
        max_pieces = [8,2,2,2,2,1][i]
        expected_shape = torch.Size([2, FENS_TO_KEEP, max_pieces])
        if tensor.shape != expected_shape:
            print(f"FAILED ctypes tensor shapes test: king bucketed PSTs[{i}] expected shape {expected_shape} but has shape {tensor.shape}")
            passed = False

    # 2. Continuous tensor
    # We will skip this, it can change too much

    # 3. Results tensor
    if get_results().shape != torch.Size([FENS_TO_KEEP]):
        print(f"FAILED ctypes tensor shapes test: results expected shape {torch.Size([FENS_TO_KEEP])} but has shape {get_results().shape}")
        passed = False

    # 4. Phases tensor
    if get_phases().shape != torch.Size([FENS_TO_KEEP, 2]):
        print(f"FAILED ctypes tensor shapes test: results expected shape {torch.Size([FENS_TO_KEEP, 2])} but has shape {get_phases().shape}")
        passed = False

    if passed:
        print("Passed ctypes tensor shapes test")
    return passed
    # TODO: Include dataset types maybe

def ctypes_dtypes_test() -> bool:
    passed = True

    # 1. King bucketed PSTs
    for i, tensor in enumerate(get_embeddings()):
        if tensor.dtype != torch.int32:
            print(f"FAILED ctypes tensor dtypes test: king bucketed PSTs[{i}] expected dtype {torch.int32} but has type {tensor.dtype}")
            passed = False

    # 2. Everything else
    for name, tensor in zip(
        ["continuous", "results", "phases"],
        [get_continuous_tensor(), get_results(), get_phases()]
    ):
        if tensor.dtype != torch.float32:
            print(f"FAILED ctypes tensor dtypes test: expected type {torch.float32} for {name} but got {tensor.dtype}")
            passed = False

    if passed:
        print("Passed ctypes tensor dtypes test")
    return passed

def null_model_test():
    passed = True
    model = Tuner()
    for param in get_detached_params(model):
        if not (param == 0).all():
            print("FAILED null model test: some parameters are not all zero")
            passed = False

    result = model(get_phases(), get_continuous_tensor(), get_embeddings())
    if not torch.all(result == 0.5):
        print("FAILED null model test: result is not all 0.5")
        print(f"result is {result}")
        passed = False
    if passed:
        print("Passed model conformable test")
    return passed

def adam_determinism_test(epochs: int = 16):
    model1 = get_model(max_epochs=epochs, optimizer = "adam", verbose=False)
    model2 = get_model(max_epochs=epochs, optimizer = "adam", verbose=False)
    deterministic = model1.get_hash() == model2.get_hash()
    print("Passed: adam is deterministic" if deterministic else "FAILED: adam is not deterministic")
    return deterministic

def lbfgs_determinism_test(epochs: int = 2):
    model1 = get_model(max_epochs=epochs, optimizer = "lbfgs", verbose=False)
    model2 = get_model(max_epochs=epochs, optimizer = "lbfgs", verbose=False)
    deterministic = model1.get_hash() == model2.get_hash()
    print("Passed: lbfgs is deterministic" if deterministic else "FAILED: lbfgs is not deterministic")
    return deterministic

def l2_changes_results_test():
    passed = True
    model1 = get_model(max_epochs=30, optimizer = "adam", verbose=False, l2_lambda=0)

    model2 = get_model(max_epochs=30, optimizer = "adam", verbose=True, l2_lambda=1e-2)
    passed &= model1.get_hash() != model2.get_hash()
    if not passed:
        print("FAILED: L2 does not change results for adam")
    model1 = get_model(max_epochs=12, optimizer = "lbfgs", verbose=False, l2_lambda=0)
    model2 = get_model(max_epochs=12, optimizer = "lbfgs", verbose=True, l2_lambda=1e-2)
    passed &= model1.get_hash() != model2.get_hash()
    if not passed:
        print("FAILED: L2 does not change results for lbfgs")
    if passed:
        print("Passed: adam and lbfgs both chanage results")
    return passed

def dimensions_l2_test():
    passed = True
    model_null = Tuner()
    if model_null.get_weird_l2_norm() != 0:
        print(f"FAILED: Null model has nonzero L2 norm of {model_null.get_weird_l2_norm()}")
    model_horiz = Tuner() # 1 horizontal row is 1s
    model_vert = Tuner() # 1 vertical column is 1s
    with torch.no_grad():
        model_horiz.embedding_layers[0].weight[0,:] += 1
        model_vert.embedding_layers[0].weight[:,0] += 1

    # print(model_horiz.embedding_layers[0].weight)
    # print(model_vert.embedding_layers[0].weight)

    horiz_l2 = model_horiz.get_weird_l2_norm()
    if horiz_l2 == 0:
        print("FAILED: L2 norm of null model with a row of 1s has zero L2 norm")
        passed = False
    else:
        print(f"Passed horizontal variance test: null model with a row of 1s has L2 norm {horiz_l2}")
    vert_l2 = model_vert.get_weird_l2_norm()
    if vert_l2 != 0:
        print("FAILED: null model with a column of 1s has L2 norm {vert_l2}")
        passed = False
    if passed:
        print("Passed dimensions l2 test")
    return passed

ALL_TESTS = [
    ctypes_shapes_test,
    ctypes_dtypes_test,
    null_model_test,
    adam_determinism_test,
    lbfgs_determinism_test,
    l2_changes_results_test,
    dimensions_l2_test
]

def run_all_tests():
    print("----------------------------")
    print("--- Tuner test functions ---")
    print("----------------------------")

    passed = True
    for test in ALL_TESTS:
        passed &= test()

    if passed:
        print("--------------------------")
        print("---- All tests passed ----")
        print("--------------------------")

def model_forward_walkthrough():
    print("Here's the continuous tensor shape:")
    print(get_continuous_tensor().shape)
    print("Here's the first 3 rows of the continuous tensor: ")
    print(get_continuous_tensor()[0:3,:])
    print("Here's the first 3 fens:")
    with open("Noah.epd", "r") as file:
        for _ in range(3):
            print(file.readline().strip())
    print("You can check that the fens match continuous features (e.g. mobility) if you like.")

    print("Now for the more difficult one: the embeddings")
    print("Right now we just have king bucketed PSTs for the embeddings.")
    print("Here's the code that generates the king bucketed PSTs:")
    print(
    """
    king_bucketed_psts = []
    for piece in range(6):
        these_psts_ = [get_king_bucketed_psts(side,piece) for side in (0,1)]
        these_psts = torch.stack(these_psts_)
        king_bucketed_psts.append(these_psts)
    return king_bucketed_psts
    """
    )
    print("So we see that king_bucketed_psts is a list of 6 tensors. What is 1 element in that list?")
    print("[get_king_bucketed_psts(side,piece) for side in (0,1)], which will then get stacked")
    print("Let's examine king bucketed PSTs for pawn and white (piece = 0, side = 0)")
    print("And let's just see the first 3 rows.")
    print(f"So the padding index is {EMBEDDING_DIMS[0]}. That means no piece.")
    print(get_king_bucketed_psts(0, 0)[0:3,:])
    print("Now let's look at the layers of the model.")
    model = get_model(optimizer = "adam", max_epochs=4, verbose = False)
    print("Let's begin with the embedding layer")
    print(f"The shape of the pawn embedding layer is {model.embedding_layers[0].weight.shape}.")
    # print(f"Here's the layer itself:\n{model.embedding_layers[0].weight}")
    print(f"The first embedding shape is {model.embedding_layers[0](get_embeddings()[0]).shape}")

    # print("LOL")
    # print(model.embedding_layers[0].weight[:,0])
    # print(model.embedding_layers[0].weight[:,1])
    # This function seems slightly overkill

all_hyperparameters = {
    "adam_standard": {
        "optimizer": "adam",
    },
    "lr=1e-4": {
        "optimizer": "adam",
        "base_lr": 1e-4
    },
    "lr=1e-2": {
        "optimizer": "adam",
        "base_lr": 1e-2
    },
    "l2=0": {
        "optimizer": "adam",
        "l2_lambda":0
    },
    "l2=1e-1": {
        "optimizer": "adam",
        "l2_lambda":1e-1
    },
    "l2=1e-3": {
        "optimizer": "adam",
        "l2_lambda":1e-3
    },
    "l2=1e-4": {
        "optimizer": "adam",
        "l2_lambda":1e-4
    },
    "l2=1e-5": {
        "optimizer": "adam",
        "l2_lambda":1e-5
    },
    "l2=1e-6": {
        "optimizer": "adam",
        "l2_lambda":1e-6
    },
    "sgd": {
        "optimizer": "sgd"
    }
}
sample_hyperparameters = {k:all_hyperparameters[k] for k in ["adam_standard", "l2=0", "l2=1e-1", "l2=1e-3", "l2=1e-4", "l2=1e-5", "l2=1e-6"]}

def compare_hyperparameters():
    run_datas : list[dict] = []
    labels : list[str] = []
    models : list[Tuner] = []
    for label in sample_hyperparameters:
        kwargs = sample_hyperparameters[label].copy()
        kwargs["max_epochs"] = 1500
        run_data = {}
        kwargs["run_data"] = run_data
        run_datas.append(run_data)
        labels.append(label)
        models.append(get_model(**kwargs))
    
    compare_loss_graph(
        losses = [run_data["val_loss"] for run_data in run_datas],
        loss_labels = labels,
        title = "Validation Loss",
        path = "graphs/compare_hyperparameters_val.png",
        start_epoch=200
    )
    compare_loss_graph(
        losses = [run_data["val_loss"] for run_data in run_datas],
        loss_labels = labels,
        title = "BCE Train Loss",
        path = "graphs/compare_hyperparameters_bce.png",
        start_epoch=200
    )
    for label, model in zip(labels, models):
        print(f"Model {label} has weight variance {model.get_weird_l2_norm().item():.4f}")

def main():
    # model_forward_walkthrough()
    # print_huge_l2_run()
    initialize_tensors()
    # print_tensor_shapes()
    check_all_hashes()
    # overwrite_all_hashes()
    # run_all_tests()
    compare_hyperparameters()
    # print_weight_decay()
    return 0

if __name__ == "__main__":
    sys.exit(main())