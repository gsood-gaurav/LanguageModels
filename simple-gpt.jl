using Printf
using Random
using Flux
# using Turing
using Functors
using CUDA
using Statistics
using LinearAlgebra
using NNlib
using ChainRules
using ChainRulesCore
using Zygote
using Distributions


# Hyperparameters
batch_size = 32 #4 #64
block_size = 8 #256
max_iters = 5000 #5000
eval_interval = 500 #500
learning_rate = 1e-3 #3e-4
eval_iters = 200
n_embd = 32
# n_embd = 384
n_head = 4
n_layer = 6
dropout = 0.2
# head_size = Int32(n_embd/n_head)
head_size = 32

# Check for CUDA
# device = CUDA.has_cuda() ? gpu : cpu
device = cpu
Random.seed!(1337)

f = open("input.txt", "r")
# Total 
text = read(f, String)

println("Length of dataset in chars: ", length(text))
println(typeof(SubString(text, 1, 1000)))
println(typeof(text[1:1000]))
println(typeof(@views text[1:1000]))

chars = sort(collect(Set(text)))
vocab_size = length(chars)

# convert list of chars to String.
# String(chars)

stoi = Dict([(k,v) for (v, k) in enumerate(chars)])
itos = Dict([(k, v) for (v, k) in pairs(stoi)])
encode(s) = [stoi[c] for c in s]
decode(s) = String([itos[i] for i in s])

# The whole text is represented as array of integers.
data = encode(text)
println(size(data))
# println(data[1:1000])
# eltype(data)

n = ceil(Int64, 0.9 * length(data))
train_data = data[1:n]
val_data = data[n+1:end]


function get_batch(split)
    data = split == "train" ? train_data : val_data
    ix = rand(1:length(data)-block_size, (batch_size,))
    x = stack([data[i:i+block_size-1] for i in ix], dims=2)
    y = stack([data[i+1:i+block_size] for i in ix], dims=2)

    return x, y  # Both x, y are of dimension (T, B)
end


function estimate_loss(model)
    out = Dict{String, Float32}()
    for split in ["train", "val"]
        losses = zeros(eval_iters)
        for k in range(1, eval_iters)
            X, Y = get_batch(split)
            loss = model(X, Y)
            losses[k] = loss
        end
        out[split] = Statistics.mean(losses)
    end

    return out
end


@kwdef struct FeedFwd
    n_embd::Int
    net = Chain(Dense(n_embd, 4*n_embd, relu),
                Dense(4*n_embd, n_embd),
                Dropout(dropout))
    
end

function (ffn::FeedFwd)(x)
    return ffn.net(x)
end

Flux.@functor FeedFwd

"""
Single Head of Self Attention
"""
@kwdef struct Head
    key = Dense(n_embd, head_size, bias=false)
    query = Dense(n_embd, head_size, bias=false)
    value = Dense(n_embd, head_size, bias=false)

    # dpout = Dropout(dropout)
end


function apply_causal_mask(attn_matrix, T)
    # tr = UpperTriangular(trues(T, T))
    # neginf = typemin(eltype(attn_matrix))
    # # println(eltype(attn_matrix))
    # attn_matrix = ifelse.(tr, attn_matrix, neginf)

    # attn_matrix = apply_causal_mask(attn_matrix, T)
    # attn_matrix = NNlib.batched_mul(permutedims(q, (2, 1, 3)), k) .* (attn_head.head_size ^ -0.5)
    mask = (1 .- reshape(triu(ones(Float32, T, T)), T, T,:)) .* (-1f20)
    # mask = (1 .- reshape(triu(ones(Float32, T, T)), Val(3))) .* (-1f20)
    # mask = device(mask)
    # attn_matrix = attn_matrix .+ mask

    # return attn_matrix
    return mask
end

@non_differentiable apply_causal_mask(::Any...)

function (attn_head::Head)(x)
    C, T, B = size(x)
    k = attn_head.key(x)
    q = attn_head.query(x)

    attn_matrix = NNlib.batched_mul(PermutedDimsArray(k, (2, 1, 3)), q) .* (head_size ^ -0.5)
    # attn_matrix = apply_causal_mask(attn_matrix, T)

    # tr = UpperTriangular(trues(T, T))
    # neginf = typemin(eltype(attn_matrix))
    # # println(eltype(attn_matrix))
    # attn_matrix = ifelse.(tr, attn_matrix, neginf)


    # mask = (1 .- reshape(triu(ones(Float32, T, T)), Val(3))) .* (-1f20)
    mask = apply_causal_mask(attn_matrix, T)
    attn_matrix = attn_matrix .+ mask
    # 
    # mask = (1 .- reshape(tril(ones(Float32, T, T)), Val(3))) .* (-1f20)
    # mask = gpu(mask)
    # attn_matrix = attn_matrix .+ mask
    # attn_matrix = causal_mask(attn_matrix, T)
    # Create multidimensional indexing
    # tr = LowerTriangular(trues(8, 8))
    # neginf = typemin(eltype(attn_matrix))
    # attn_matrix = ifelse.(tr, attn_matrix, neginf)
    # tr = UpperTriangular(trues(T, T))
    # tr[diagind(tr)] .= 0
	# mask = repeat(tr[:,:,:], outer=(1,1,B))

    # attn_matrix[mask] .= -Inf32
	attn_matrix = softmax(attn_matrix, dims=1)
    # attn_matrix = attn_head.dpout(attn_matrix)
	v = attn_head.value(x)
	out = NNlib.batched_mul(v, attn_matrix)

    return out
end

Flux.@functor Head

@kwdef struct MHAtten
    num_heads::Int
    head_size::Int
    attn_heads = [Head(head_size=head_size) for _ in range(1, num_heads)]
    proj = Dense(n_embd, n_embd)
    dpout = Dropout(dropout)
end

function (mha::MHAtten)(x)
    out = cat([hd(x) for hd in mha.attn_heads]...; dims=1)
    out = mha.proj(out)
    out = mha.dpout(out)

    return out
end

Flux.@functor MHAtten

struct BigramLanguageModel
    layers::NamedTuple
    # function BigramLanguageModel(vocab_size=65, batch_size=32, block_size=8)
    #     return new((emb_layer=Flux.Embedding(vocab_size, vocab_size),))
    # end
end


@kwdef struct Block
    n_embd::Int
    n_head::Int
    sa = MHAtten(num_heads=n_head, head_size=Int32(n_embd/n_head))
    ffn = FeedFwd(n_embd=n_embd)
    ln1 = LayerNorm(n_embd)
    ln2 = LayerNorm(n_embd)
end


function (decode::Block)(x)
    x = x + decode.sa(decode.ln1(x))
    x = x + decode.ffn(decode.ln2(x))

    return x
end

Flux.@functor Block
function BigramLanguageModel()
# function BigramLanguageModel(vocab_size=vocab_size, batch_size=batch_size, block_size=block_size, n_embd=n_embd, head_size=head_size)
    return BigramLanguageModel((
        emb_layer = Flux.Embedding(vocab_size, n_embd),
        position_emb_table = Flux.Embedding(block_size, n_embd),
        attn_layer = Head(),
        # mha_layer = MHAtten(num_heads=4, head_size=Int32(n_embd/4)),
        # ffn_layer = FeedFwd(n_embd=n_embd),
        # blocks = Chain([Block(n_embd=n_embd, n_head=n_head) for _ in range(1, n_layer)]...),
        # blocks = Chain(
        #     Block(n_embd=n_embd, n_head=4),
        #     Block(n_embd=n_embd, n_head=4),
        #     Block(n_embd=n_embd, n_head=4),
        #     Block(n_embd=n_embd, n_head=4),
            
        # ),
        # ln_b = LayerNorm(n_embd),
        lm_head = Flux.Dense(n_embd, vocab_size),
        ))
end

@functor BigramLanguageModel


function get_one_hot_encoding(labels, vocabulary)
    return Flux.onehotbatch(collect(labels), 1:vocabulary) |> device  # (vocab_size, T, B)
end

@non_differentiable get_one_hot_encoding(::Any...)

function (m::BigramLanguageModel)(idx, targets=nothing)

    # idx and targets are both (T, B) tensors.
    T, B = size(idx)
    # x = m.layers.emb_layer(idx) # (C, T, B)
    tok_emb = m.layers.emb_layer(idx) # (C, T, B)
    pos_emb = m.layers.position_emb_table(1:T)  # (C, T)
    x = tok_emb .+ pos_emb  # (C, T, B)
    x = m.layers.attn_layer(x)
    # x = m.layers.mha_layer(x)
    # x = m.layers.attn_layer(x)
    # x = m.layers.ffn_layer(x)  # It at token level (per token)
    # x = m.layers.blocks(x)
    # x = m.layers.ln_b(x)
    logits = m.layers.lm_head(x)  # (vocab_size, T, B)

    if targets === nothing
        return logits

    else
        # ylabels = get_one_hot_encoding(targets, vocab_size)
        ylabels = Flux.onehotbatch(collect(targets), 1:vocab_size)  # (vocab_size, T, B)
        # ylabels = gpu(ylabels)
        # logits = cpu(logits)
        loss = Flux.logitcrossentropy(logits, ylabels)
        return loss
    end

    # return loss, logits
end

m = BigramLanguageModel() |> device
# loss2 = m2(xb, yb)
opt = ADAM(learning_rate)
ps = Flux.params(m)
# println(ps)
# loss = 0.0
for iter in range(0, max_iters)

    if iter % eval_interval == 0
        losses = estimate_loss(m)
        @printf "step %d: train loss %.4f, val loss %.4f\n" iter losses["train"] losses["val"]
    end

    xb, yb = get_batch("train")
    xb = device(xb)
    yb = device(yb)
    loss, grad = Flux.withgradient(ps) do
        m(xb, yb)
    end
    Flux.Optimise.update!(opt, ps, grad)
    # @show loss
end
# @show loss3

function generate(idx, max_tokens=1000)
    buf = Array{Int, 1}()
    for _ in range(1, max_tokens)
        push!(buf, idx)
        logits = m(reshape(last(buf, block_size), :, 1))
        logits = logits[:, end, :]
        prob = softmax(logits)
        prob = cpu(prob)
        # println(prob)
        # println(size(prob))
        categorical = Categorical(reshape(prob, :))
        idx = rand(categorical)
        print(itos[idx])
        # push!(buf, idx)
    end
    # print(buf)
end

generate(1)