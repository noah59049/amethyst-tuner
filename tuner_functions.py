#################
#### Imports ####
#################
import time
import numpy as np
import torch
import torch.nn as nn
import ctypes
from ctypes import POINTER, c_int, c_float
import functools
import io
import hashlib

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

FENS_TO_KEEP = 7153281
# Max possible value for FENS_TO_KEEP is 7153281 for Noah.epd
# Higher values will give an error because we don't have enough fens
# Set to something like 100000 for debugging purposes

@functools.cache
def get_results():
    results = (c_float * FENS_TO_KEEP)()
    libtuner.getResults(results,
                       c_int(FENS_TO_KEEP))
    results = np.ctypeslib.as_array(results, shape = (FENS_TO_KEEP,))
    results = torch.tensor(results)
    return results

@functools.cache
def get_phases():
    phases = (c_int * FENS_TO_KEEP)()
    libtuner.getPhases(phases,
                       c_int(FENS_TO_KEEP))
    phases = np.ctypeslib.as_array(phases, shape = (FENS_TO_KEEP,))
    phases = torch.tensor(phases)
    phases = torch.stack([phases, 24 - phases])
    phases = torch.transpose(phases, 0, 1).float()
    return phases

@functools.cache # I don't think we really need this
def get_king_squares(side):
    kingSquares = (c_int * FENS_TO_KEEP)()
    libtuner.getKingSquares(c_int(side), kingSquares, c_int(FENS_TO_KEEP))
    kingSquares = np.ctypeslib.as_array(kingSquares, shape=(FENS_TO_KEEP,))
    return kingSquares.reshape(-1,1)

@functools.cache 
def get_mobility(side):
    mobility = (c_int * (6 * FENS_TO_KEEP))()
    libtuner.getMobility(c_int(side), mobility, c_int(FENS_TO_KEEP))
    mobility = np.ctypeslib.as_array(mobility, shape=(6 * FENS_TO_KEEP,))
    return mobility.reshape(-1,6)
    # We will convert to torch tensor and to float when we stack all the tensors together

@functools.cache
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

@functools.cache
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
def get_all_psts():
    return [torch.stack([get_psts(0,piece), get_psts(1, piece)]) for piece in range(6)]
@functools.cache
def get_all_king_bucketed_psts():
    return [torch.stack([get_king_bucketed_psts(0,piece), get_king_bucketed_psts(1, piece)]) for piece in range(6)]

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
def get_continuous_tensor():
    return get_continuous_tensor_(CONTINUOUS_FUNCTIONS)

###################################
#### Initialization of tensors ####
###################################

def initialize_tensors():
    t0 = time.time()
    ONE_HOT_TENSORS = get_all_king_bucketed_psts()
    t1 = time.time()
    print(f"Made embeddings in {t1-t0:.2f} seconds")

    t0 = time.time()
    CONTINUOUS_TENSOR = get_continuous_tensor()
    t1 = time.time()
    print(f"Made continuous tensor in {t1-t0:.2f} seconds")


    t0 = time.time()
    RESULTS = get_results()
    t1 = time.time()
    print(f"Made results tensor in {t1-t0:.2f} seconds")

    t0 = time.time()
    PHASES = get_phases()
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
        max_delta = torch.max(deltas)
        overall_max_delta = max(max_delta, overall_max_delta)
        
    return overall_max_delta

def get_model(model = None, optimizer = "lbfgs", max_epochs = 3200):
    # 1. Random stuff we need
    t0 = time.time()
    if model is None:
        model = Tuner()
    optimizer_str = optimizer
    
    # 2. Optimizer, loss function, closure for LBFGS
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
        y_pred = model(get_phases(), get_continuous_tensor(), get_all_king_bucketed_psts())
        loss = loss_function(y_pred, get_results())
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
        y_pred = model(get_phases(), get_continuous_tensor(), get_all_king_bucketed_psts())
        loss = loss_function(y_pred, get_results())
        loss.backward()
        if optimizer_str == "lbfgs":
            optimizer.step(closure)
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

        message = f"epoch {epoch}; t={int(t1-t0)}; loss={loss.item():.4f}"
        if check_max_delta:
            message += f"; max(∆w) = {max_delta:.4f}"
            
        # 4C. Stopping criteria
        # If none of the params have changed more than 0.5 since last time we checked, we break out of the loop.
        if max_delta < STOPPING_DELTA:
            break

    # 5. Return the model
    model.round_weights()
    return model

#########################################
#### Printing weights, saving evals  ####
#########################################

# I could indent these, but why bother
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

def print_king_bucketed_psts(model: Tuner):
    params = model.embedding_layers
    print(f"    constexpr std::array<std::array<std::array<packed_eval_t, 64>, 6>, {NUM_KING_BUCKETS}> psts = " + "{{")
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

eval_t_name = "packed_eval_t"
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
    evals = model.forward_as_in_engine(get_phases(), get_continuous_tensor(), get_all_king_bucketed_psts())
    with open(filename, "w") as file:
        for eval_ in evals:
            file.write(str(int(eval_.item())))
            file.write('\n')
            
    print(f"len(evals) is {len(evals)}")
    file.close()

###############
#### Tests ####
###############

def print_tensor_shapes():
    print(f"Embeddings shapes: {[t.shape for t in get_all_king_bucketed_psts()]}")
    print(f"Continuous tensor shape: {get_continuous_tensor().shape}")
    print(f"Results tensor shape: {get_results().shape}")
    print(f"Phases tensor shape: {get_phases().shape}")

def print_tensor_dtypes():
    print(f"Embeddings dtypes: {[t.dtype for t in get_all_king_bucketed_psts()]}")
    print(f"Continuous tensor dtype: {get_continuous_tensor().dtype}")
    print(f"Results tensor dtype: {get_results().dtype}")
    print(f"Phases tensor dtype: {get_phases().dtype}")

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