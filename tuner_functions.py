################
#### Cell 1 ####
################
import time
import numpy as np
import torch
import torch.nn as nn
import ctypes
from ctypes import POINTER, c_int, c_float

################
#### Cell 2 ####
################
NUM_FENS = 7153281

BLACK_KING_BUCKETS = [0, 0, 0, 0, 1, 1, 1, 1,
                      0, 0, 0, 0, 1, 1, 1, 1,
                      0, 0, 0, 0, 1, 1, 1, 1,
                      0, 0, 0, 0, 1, 1, 1, 1,
                      0, 0, 0, 0, 1, 1, 1, 1,
                      0, 0, 0, 0, 1, 1, 1, 1,
                      0, 0, 0, 0, 1, 1, 1, 1,
                      0, 0, 0, 0, 1, 1, 1, 1]

NUM_KING_BUCKETS = 2

# Step 1: Load the library
lib_path = "libamethyst_tuner.dylib"
libtuner = ctypes.cdll.LoadLibrary(lib_path)

# Step 2: Set the argtypes of all functions from the library
libtuner.getResults.argtypes = [POINTER(c_float), c_int]
libtuner.getPhases.argtypes = [POINTER(c_int), c_int]
libtuner.getKingSquares.argtypes = [c_int, POINTER(c_int), c_int]
libtuner.getMobility.argtypes = [c_int, POINTER(c_int), c_int]
libtuner.getPSTs.argtypes = [c_int, c_int, c_int, POINTER(c_int), c_int]

def get_results():
    results = (c_float * NUM_FENS)()
    libtuner.getResults(results,
                       c_int(NUM_FENS))
    results = np.ctypeslib.as_array(results, shape = (NUM_FENS,))
    results = torch.tensor(results)
    return results

def get_phases():
    phases = (c_int * NUM_FENS)()
    libtuner.getPhases(phases,
                       c_int(NUM_FENS))
    phases = np.ctypeslib.as_array(phases, shape = (NUM_FENS,))
    phases = torch.tensor(phases)
    phases = torch.stack([phases, 24 - phases])
    phases = torch.transpose(phases, 0, 1).float()
    return phases

def get_king_squares(side):
    kingSquares = (c_int * NUM_FENS)()
    libtuner.getKingSquares(c_int(side), kingSquares, c_int(NUM_FENS))
    kingSquares = np.ctypeslib.as_array(kingSquares, shape=(NUM_FENS,))
    return kingSquares.reshape(-1,1)
    
def get_mobility(side):
    mobility = (c_int * (6 * NUM_FENS))()
    libtuner.getMobility(c_int(side), mobility, c_int(NUM_FENS))
    mobility = np.ctypeslib.as_array(mobility, shape=(6 * NUM_FENS,))
    return mobility.reshape(-1,6)

def get_psts(side, piece):
    maxPieces = [8,2,2,2,2,1][piece]
    psts = (c_int * (maxPieces * NUM_FENS))()
    libtuner.getPSTs(c_int(side),      # side
                     c_int(piece),     # piece
                     c_int(maxPieces), # maxPieces
                     psts,             # arr
                     c_int(NUM_FENS))      # length
    
    psts = np.ctypeslib.as_array(psts, shape = (maxPieces * NUM_FENS,))
    psts = psts.reshape(-1,maxPieces)
    psts.sort()
    return torch.tensor(psts)

def get_king_bucketed_psts(side, piece):
    psts = get_psts(side, piece)
    king_squares = get_king_squares(side).reshape(-1)
    cap = torch.tensor(64 * NUM_KING_BUCKETS)
    
    king_buckets = torch.tensor([BLACK_KING_BUCKETS[ksq ^ 56] for ksq in king_squares]).reshape(-1, 1)
    
    psts = torch.where(psts == 64, torch.tensor(10000), psts)
    psts += 64 * king_buckets
    psts = torch.minimum(psts, cap)
    return psts

################
#### Cell 3 ####
################
def get_all_psts():
    return [torch.stack([get_psts(0,piece), get_psts(1, piece)]) for piece in range(6)]
def get_all_king_bucketed_psts():
    return [torch.stack([get_king_bucketed_psts(0,piece), get_king_bucketed_psts(1, piece)]) for piece in range(6)]

################
#### Cell 4 ####    not actually
################

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

def get_continuous_tensor(features):
    t0 = time.time()
    continuous_tensors = []
    for feature in features:
        tensor = feature.get_diff()
        continuous_tensors.append(tensor)
    result = torch.hstack(continuous_tensors)
    t1 = time.time()
    print(f"Made continuous tensor in {t1-t0:.2f} seconds")
    return result

################
#### Cell 4 ####    not actually
################
t0 = time.time()
ONE_HOT_TENSORS = get_all_king_bucketed_psts()
t1 = time.time()
print(f"Made embeddings in {t1-t0:.2f} seconds")

CONTINUOUS_TENSOR = get_continuous_tensor(CONTINUOUS_FUNCTIONS)

t0 = time.time()
RESULTS = get_results()
t1 = time.time()
print(f"Made results tensor in {t1-t0:.2f} seconds")

t0 = time.time()
PHASES = get_phases()
t1 = time.time()
print(f"Made phase tensor in {t1-t0:.2f} seconds")

EMBEDDING_DIMS = [128, 128, 128, 128, 128, 128] # I don't remember what this does
SCALING_TERM = 28 * 400 / 2.1 
# Yes that should be a 24 instead of 28 and that's on my to do list to change
# Unfortunately it's not as simple as "just change it"
# The "28" typo effectively uses a different scaling term than 400
# This changes the weights of everything

class Tuner(nn.Module):
    def __init__(self):
        super(Tuner, self).__init__()
        self.continuous_layer = nn.Linear(CONTINUOUS_TENSOR.shape[1], 2, bias = False)
        self.embedding_layers = [nn.Embedding(dim + 1, 2, padding_idx = dim) for dim in EMBEDDING_DIMS]
        with torch.no_grad():
            self.continuous_layer.weight.zero_()
            for i, layer in enumerate(self.embedding_layers):
                layer.weight.zero_()
                self.register_parameter(f"embedding{i}", layer.weight)
                
    def get_logits(self, layer, phases, embedding_tensor):
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

def get_detached_params(model):
    return [param.detach().clone() for param in model.parameters()]

def get_max_delta(params1, params2):
    overall_max_delta = 0
    for param1, param2 in zip(params1, params2):
        deltas = torch.abs(param2 - param1)
        max_delta = torch.max(deltas)
        overall_max_delta = max(max_delta, overall_max_delta)
        
    return overall_max_delta

def get_model(model = None, optimizer = "lbfgs", max_epochs = 3200):
    # 1. Random stuff we need
    if model is None:
        model = Tuner()
    t0 = time.time()
    optimizer_str = optimizer
    
    # 2. The actual stuff needed for training
    loss_function = nn.BCELoss()
    if optimizer_str == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters())
    elif optimizer_str == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = 10)
    else:
        assert False # Optimizer must be "lbfgs" or "adam"
    
    def closure():
        # This is the closure required for LBFGS
        optimizer.zero_grad()
        y_pred = model(PHASES, CONTINUOUS_TENSOR, ONE_HOT_TENSORS)
        loss = loss_function(y_pred, RESULTS)
        loss.backward()
        return loss
    
    # 3. Stopping criteria
    STOPPING_DELTA = 0.5
    previous_params = None
    current_params = None
    
    # 4. Start the training loop
    for epoch in range(max_epochs):
        # 4A. The actual important training
        y_pred = model(PHASES, CONTINUOUS_TENSOR, ONE_HOT_TENSORS)
        loss = loss_function(y_pred, RESULTS)
        loss.backward()
        if optimizer_str == "lbfgs":
            optimizer.step(closure)
        else:
            optimizer.step()
        optimizer.zero_grad()

        # 4B. Printouts
        t1 = time.time()
        if True:
            print(f"epoch {epoch+1}; loss = {loss.item():.4f}; total time = {int(t1-t0)}")
            
        # 4C. Stopping criteria
        # If, in the past 10 epochs (or 1 epoch if using LBFGS), none of the params have changed by more than 0.5, we are done.
        if optimizer_str == "lbfgs" or (epoch % 10 == 9):
            current_params = get_detached_params(model)
            if previous_params is not None:
                max_delta = get_max_delta(previous_params, current_params)
                print(f"Max parameter weight change is {max_delta:.4f}")
                if max_delta < STOPPING_DELTA:
                    break

            previous_params = current_params

    model.round_weights()
    return model

def print_psts(params):
    print("constexpr std::array<std::array<packed_eval_t, 64>, 6> psts = {{")
    for piece_type in range(6):
        print("{")

        # Determine how much whitespace we need
        l1 = len(str(torch.min(params[piece_type].weight.data.int()).item()))
        l2 = len(str(torch.max(params[piece_type].weight.data.int()).item()))
        l = max(l1, l2)

        for i, thing in enumerate(params[piece_type].weight.data.int()[0:64,:]):
            print(f"S({thing[0]:{l}},{thing[1]:{l}}), ", end = "\n" * (i % 8 - 6))
        print("},")

    print("}};")

def print_king_bucketed_psts(params):
    print(f"constexpr std::array<std::array<std::array<packed_eval_t, 64>, 6>, {NUM_KING_BUCKETS}> psts = " + "{{")
    for bucket in range(NUM_KING_BUCKETS):
        idx = bucket * 64
        print("{{")
        for piece_type in range(6):
            print("{")

            # Determine how much whitespace we need
            l1 = len(str(torch.min(params[piece_type].weight.data.int()).item()))
            l2 = len(str(torch.max(params[piece_type].weight.data.int()).item()))
            l = max(l1, l2)

            for i, thing in enumerate(params[piece_type].weight.data.int()[idx:idx+64,:]):
                print(f"S({thing[0]:{l}},{thing[1]:{l}}), ", end = "\n" * (i % 8 - 6))
            print("},")
        print("}},")
    print("}};")

eval_t_name = "packed_eval_t"
def print_continuous_weights(model):
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

# IIRC this function is used to make sure the engine and tuner evals match exactly
def save_eval_to_file(model, filename):
    evals = model.forward_as_in_engine(PHASES, CONTINUOUS_TENSOR, ONE_HOT_TENSORS)
    with open(filename, "w") as file:
        for eval_ in evals:
            file.write(str(int(eval_.item())))
            file.write('\n')
            
    print(f"len(evals) is {len(evals)}")
    file.close()

#####################
#### Example run ####
#####################
"""
model1 = Tuner()
with torch.no_grad():
    for piece in range(6):
        for phase in range(2):
            model1.embedding_layers[piece].weight[:,phase] = MATERIAL_START_WEIGHTS[phase,piece]
            # IMPORTANT: We need to set the weights for the padding index to 0
            model1.embedding_layers[piece].weight[NUM_KING_BUCKETS * 64] = 0

model2 = get_model(model1, optimizer = "lbfgs", max_epochs = 100) # This apparently got stuck

model3 = get_model(model1, optimizer = "adam") # This took like 3 hours

tuner4 = get_model(optimizer = "lbfgs") # This one doesn't get stuck

print_continuous_weights(model2) # Wait why model2, it got stuck 
print_king_bucketed_psts(model2.embedding_layers) # why model2 again
"""