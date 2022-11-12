import DataFrames
import CSV

@enum Activation sigmoid=1 relu=2 linear=3 tanh=4

# We parse the parameters

if length(ARGS) != 6
    throw(ArgumentError("Invalid number of parameters."))
end

dataset_path = ARGS[1]

layers = let expr = Meta.parse(ARGS[2])
    @assert expr.head == :vcat
    Int64.(expr.args)
end
fact_str = ARGS[3]
learning_rate = parse(Float64, ARGS[4])
momentum = parse(Float64, ARGS[5])
epochs = parse(Int64, ARGS[6])

println("PARAMETERS: normalized_dataset=\"", dataset_path, "\" layers=", layers, " fact=", fact_str, " learning_rate=", learning_rate, " momentum=", momentum, " epochs=", epochs)

print("Reading data...")
train_df = CSV.read(dataset_path * "/train.txt", DataFrames.DataFrame, delim='	')
test_df = CSV.read(dataset_path * "/test.txt", DataFrames.DataFrame, delim='	')
println("done.")

struct NeuralNet
    L::Int64                              # number of layers
    n::Vector{Int64}                      # sizes of layers
    h::Vector{Vector{Float64}}            # units field
    xi::Vector{Vector{Float64}}           # units activation
    w::Vector{Array{Float64,2}}           # weights
    theta::Vector{Vector{Float64}}        # thresholds
    delta::Array{Array{Float64}}          # propagation error
    d_w::Vector{Array{Float64,2}}         # weights delta
    d_theta::Vector{Vector{Float64}}      # thresholds delta
    d_w_prev::Vector{Array{Float64,2}}    # previous weights delta
    d_theta_prev::Vector{Vector{Float64}} # previous thresholds delta
    fact::Activation                      # activation function
end

function NeuralNet(layers::Vector{Int64})
    L = length(layers)
    n = copy(layers)

    h = Vector{Float64}[]
    xi = Vector{Float64}[]
    theta = Vector{Float64}[]
    d_theta = Vector{Float64}[]
    delta = Array{Float64}[]
    for l in 1:L
        push!(h, zeros(layers[l]))
        push!(xi, zeros(layers[l]))
        push!(theta, rand(layers[l]))                    # random, but should have also negative values
        push!(d_theta, zeros(layers[l]))
        push!(delta, zeros(layers[l]))
    end

    # Initialize weights

    w = Array{Float64,2}[]
    d_w = Array{Float64,2}[]
    push!(w, zeros(1, 1))                          # unused, but needed to ensure w[2] refers to weights between the first two layers
    push!(d_w, zeros(1, 1))
    for l in 2:L
        push!(w, rand(layers[l], layers[l - 1]))     # random, but should have also negative values
        push!(d_w, zeros(layers[l], layers[l - 1]))
    end

    d_w_prev = d_w
    d_theta_prev = d_theta
    
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

    return NeuralNet(L, n, h, xi, w, theta, delta, d_w, d_theta, d_w_prev, d_theta_prev, fact)
end

function sigmoid_func(h::Float64)::Float64
    return 1 / (1 + exp(-h))
end

function sigmoid_derivative(x::Vector{Float64})::Vector{Float64}
    return x .* (1 .- x)
end

function sigmoid_derivative(x::Float64)::Float64
    return x * (1 - x)
end

#FIXME: outputs [NaN] starting from some epoch
function feed_forward!(nn::NeuralNet, x_in::Vector{Float64})::Vector{Float64}
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
            nn.xi[l][i] = (nn.fact === sigmoid::Activation ? sigmoid_func(h) : throw(ArgumentError("This activation function is not implemented")))
        end
    end

    # copy activation in output layer as output, Eq. (9)
    return copy(nn.xi[nn.L])
end

function transfer(fact::Activation, activation_vector::Vector{Float64})::Vector{Float64}
    return (fact === sigmoid::Activation ? sigmoid_derivative(activation_vector) : throw(ArgumentError("This activation function is not implemented")))
end

function transfer(fact::Activation, activation_scalar::Float64)::Float64
    return (fact === sigmoid::Activation ? sigmoid_derivative(activation_scalar) : throw(ArgumentError("This activation function is not implemented")))
end

function back_propagation!(nn::NeuralNet, y::Vector{Float64}, z::Float64)
    for i in nn.n[nn.L]
        nn.delta[nn.L][i] = (sigmoid_func(nn.h[nn.L][i]) * (1 - sigmoid_func(nn.h[nn.L][i]))) * (y[i] - z[i]) # Eq. (11)
    end

    # back-propagation of input pattern Eq. (12)
    for l in nn.L:-1:2
        for j in 1:nn.n[l - 1]
            error = 0
            for i in 1:nn.n[l]
                error += nn.delta[l][i] * nn.w[l][i, j]
            end
            nn.delta[l-1][j] =  (sigmoid_func(nn.h[l-1][j]) * (1 - sigmoid_func(nn.h[l-1][j]))) * error
        end
    end
end

function update_weights_and_thresholds!(nn::NeuralNet, learning_rate::Float64, momentum::Float64)
    # Eq. 14
    for l in 2:nn.L
        for i in 1:nn.n[l]
            for j in 1:nn.n[l - 1]
                nn.d_w[l][i, j] = -learning_rate * nn.delta[l][i] * nn.xi[l-1][j] + momentum * nn.d_w_prev[l][i, j]
                nn.w[l][i, j] += nn.d_w[l][i, j]
                nn.d_w_prev[l][i, j] = nn.d_w[l][i, j]
            end 
        end
    end
    for l in 2:nn.L
        for i in 1:nn.n[l]
            nn.d_theta[l][i] = learning_rate * nn.delta[l][i] + momentum * nn.d_theta_prev[l][i]
            nn.theta[l][i] += nn.d_theta[l][i]
            nn.d_theta_prev[l][i] = nn.d_theta[l][i]
        end
    end
end

function train(nn::NeuralNet, x_in::Vector{Vector{Float64}}, desired_outputs::Vector{Float64}, learning_rate::Float64, momentum::Float64, epochs::Int64)
    for _ in 1:epochs
        for pattern in 1:length(x_in)
            y = feed_forward!(nn, x_in[pattern])
            back_propagation!(nn, y, desired_outputs[pattern])
            update_weights_and_thresholds!(nn, learning_rate, momentum)
        end
        #TODO: Feed−forward all training patterns and calculate their prediction quadratic error
        #TODO: Better name
        #sum = 0
        #for pattern in 1:length(x_in)
        #    what is m? basicly amount of rows in input file, right?
        #    sum += sqrt(y-desired_outputs)
        #end
        #training_prediction_quadratic_error = 0.5 * sum
        #
        #TODO: Feed−forward all validation patterns and calculate their prediction quadratic error
    end
end

function predict(nn::NeuralNet, x::Vector{Vector{Float64}})::Vector{Vector{Float64}}
    y = Vector{Vector{Float64}}()
    for pattern in 1:length(x)
        push!(y, feed_forward!(nn, x[pattern]))
    end
    return y
end

# We launch the training

print("Creating the Neural Network and initializing weights...")
nn = NeuralNet(layers)
y_out = zeros(nn.n[nn.L])
println("done.")

print("Setting up the input train data...")
x_in = Matrix{Float64}(train_df[:, 1:end-1])
x_in = [x_in[i, :] for i in 1:size(x_in, 1)]
x_desired_outputs = Vector{Float64}(train_df[:, end])
println("done.")

print("Training...")
train(nn, x_in, x_desired_outputs, learning_rate, momentum, epochs)
println("done.")

#println("x_desired_outputs=", x_desired_outputs)

# We launch the prediction

pred_in = Matrix{Float64}(test_df[:, 1:end-1])
pred_in = [pred_in[i, :] for i in 1:size(pred_in, 1)]

print("Predicting...")
predicted = predict(nn, pred_in)
println("done.")
println("Predicted values (last column):")
println(predicted)

print("Calculating accuracy...")
correct = 0
for i in 1:DataFrames.nrow(test_df)
    if predicted[i] == test_df[:, end]
        global correct += 1
    end
end
accuracy = correct / DataFrames.nrow(test_df)
println("Accuracy: ", accuracy*100, "%")