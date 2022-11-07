import DataFrames
import CSV
import Random

@enum Activation sigmoid=1 relu=2 linear=3 tanh=4

function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    Random.shuffle!(ids)
    sel = ids .<= DataFrames.nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end

# We parse the parameters

if length(ARGS) != 5
    throw(ArgumentError("Invalid number of parameters."))
end

dataset_file = ARGS[1]

layers = let expr = Meta.parse(ARGS[2])
    @assert expr.head == :vcat
    Int64.(expr.args)
end
fact_str = ARGS[3]
fact::Activation = if fact_str == "sigmoid"
    sigmoid::Activation
elseif fact_str == "relu" 
    relu::Activation
elseif fact_str == "linear"
    linear::Activation
elseif fact_str == "tanh"
    tanh::Activation
else
    throw(ArgumentError("This activation function is not recognized"))
end
learning_rate = parse(Float64, ARGS[4])
epochs = parse(Int64, ARGS[5])
# momentum = parse(Float64, ARGS[6])

println("PARAMETERS: normalized_dataset=\"", dataset_file, "\" layers=", layers, " fact=", fact, " learning_rate=", learning_rate, " epochs=", epochs)

print("Reading data...")
df = CSV.read(dataset_file, DataFrames.DataFrame, delim='	')
println("done.")
println("Full data:")
println(df)

print("Splitting data...")
train_df, test_df = splitdf(df, 0.8)
println("done.")
println("Train data:")
println(train_df)
println("Test data:")
println(test_df)

print("Removing target column for the test data...")
test_df = DataFrames.select(test_df, DataFrames.Not(:pop))
println("done.")
println("Final test data:")
println(test_df)

struct NeuralNet
    L::Int64                        # number of layers
    n::Vector{Int64}                # sizes of layers
    h::Vector{Vector{Float64}}      # units field
    xi::Vector{Vector{Float64}}     # units activation
    w::Vector{Array{Float64,2}}     # weights
    theta::Vector{Vector{Float64}}  # thresholds
end

function NeuralNet(layers::Vector{Int64})
    L = length(layers)
    n = copy(layers)

    h = Vector{Float64}[]
    xi = Vector{Float64}[]
    theta = Vector{Float64}[]
    for l in 1:L
        push!(h, zeros(layers[l]))
        push!(xi, zeros(layers[l]))
        push!(theta, rand(layers[l]))                    # random, but should have also negative values
    end

    # Initialize weights

    w = Array{Float64,2}[]
    push!(w, zeros(1, 1))                          # unused, but needed to ensure w[2] refers to weights between the first two layers
    for l in 2:L
        push!(w, rand(layers[l], layers[l - 1]))     # random, but should have also negative values
    end

    return NeuralNet(L, n, h, xi, w, theta)
end

function sigmoid_func(h::Float64)::Float64
    return 1 / (1 + exp(-h))
end

function sigmoid_derivative(x::Vector{Float64})::Vector{Float64}
    return x .* (1 .- x)
end

function feed_forward!(nn::NeuralNet, x_in::Vector{Float64}, y_out::Vector{Float64})
    # copy input to first layer, Eq. (6)
    nn.xi[1] .= x_in
    
    # feed-forward of input pattern
    for l in 2:nn.L
        for i in 1:nn.n[l]
            # calculate input field to unit i in layer l, Eq. (8)
            h = -nn.theta[l][i]
            for j in 1:nn.n[l - 1]
                h += nn.w[l][i, j] * nn.xi[l - 1][j]
            end
            # save field and calculate activation, Eq. (7)
            nn.h[l][i] = h
            nn.xi[l][i] = (fact === sigmoid::Activation ? sigmoid_func(h) : throw(ArgumentError("This activation function is not implemented")))
        end
    end

    # copy activation in output layer as output, Eq. (9)
    y_out .= nn.xi[nn.L]
end

function transfer(activation_vector::Vector{Float64})::Vector{Float64}
    return (fact === sigmoid::Activation ? sigmoid_derivative(activation_vector) : throw(ArgumentError("This activation function is not implemented")))
end

# not sure this works
function update_weights!(weights::Vector{Array{Float64,2}}, d_theta::Vector{Vector{Float64}}, learning_rate::Float64)
    #TODO
end

# not sure this works
function back_propagation(weights::Vector{Array{Float64,2}}, activations::Vector{Vector{Float64}}, x::Vector{Float64}, y::Vector{Float64})::Array{Float64,2}
    #TODO
end

# not sure this works
function train(nn::NeuralNet, x::Vector{Float64}, y::Vector{Float64}, learning_rate::Float64, epochs::Int64)
    #TODO
end

# not sure this works
function predict(nn::NeuralNet, x::Vector{Float64}, y::Vector{Float64})::Vector{Float64}
    feed_forward!(nn, x, y)
    return y
end

# We launch the training

print("Creating the Neural Network and initializing weights...")
nn = NeuralNet(layers)
y_out = zeros(nn.n[nn.L])
println("done.")

println("-------------------------------------------")

println("nn.L=", nn.L)
println("nn.n=", nn.n)

println("nn.xi=", nn.xi)
println("nn.xi[1]=", nn.xi[1])
println("nn.xi[2]=", nn.xi[2])


println("nn.w=", nn.w)
println("nn.w[2]=", nn.w[2])

println("-------------------------------------------")

print("Setting up the input data...")
x_in = rand(nn.n[1])                        # INPUT DATA
println("done.")

print("Training...")
train(nn, x_in, y_out, learning_rate, epochs)
println("done.")

println("y_out=", y_out)

print("Predicting...")
predicted = predict(nn, x_in, y_out)
println("done.")
println("Predicted `pop` values:")
println(predicted)

#print("Calculating accuracy...")
#correct = 0
#for i in 1:DataFrames.nrow(test_df)
#    if predicted[i] == test_df[i, :pop]
#        correct += 1
#    end
#end
#accuracy = correct / DataFrames.nrow(test_df)
#println("Accuracy: ", accuracy)