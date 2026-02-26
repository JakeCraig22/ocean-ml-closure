using Flux, NCDatasets, Statistics, Dates

#Load Data
file = "data\\era5_t2_199301-200508_mon.nc"
ds = NCDataset(file)
t2m = ds["t2"][500:600, 50:150, :]  # Region select
temp = Float32[mean(t2m[:, :, t]) for t in 1:size(t2m, 3)]
close(ds)

#Normalize
μ, σ = mean(temp), std(temp)
temp_norm = (temp .- μ) ./ σ

#Add Solar Term Features
seq_len = 12  # Use past 12 months
n_samples = length(temp_norm) - seq_len

# Year cycle encoding: sin(2π·t/12), cos(2π·t/12)
function add_seasonal_features(n)
    features = zeros(Float32, 4, seq_len, n)
    for i in 1:n
        t = (i:i+seq_len-1) .% 12
        # 1 year cycle
        features[1, :, i] = sin.(2π .* t ./ 12)
        features[2, :, i] = cos.(2π .* t ./ 12)
        # half year cycle
        features[3, :, i] = sin.(4π .* t ./ 12)
        features[4, :, i] = cos.(4π .* t ./ 12)
    end
    return features
end

X_season = add_seasonal_features(n_samples)
X_temp = zeros(Float32, 1, seq_len, n_samples)
Y = zeros(Float32, 1, n_samples)

for i in 1:n_samples
    X_temp[1, :, i] = temp_norm[i:i+seq_len-1]
    Y[1, i] = temp_norm[i+seq_len]
end

# Combine: temp + seasonal features
X = cat(X_temp, X_season, dims=1)  # Shape: (3, seq_len, n_samples)

#Split Data
n_train = Int(0.8 * n_samples)
X_train, Y_train = X[:, :, 1:n_train], Y[:, 1:n_train]
X_val, Y_val = X[:, :, n_train+1:end], Y[:, n_train+1:end]

#Build Model
model = Chain(
    LSTM(5 => 64),        # Input: 5 channels (temp + 4 seasonal), Output: 64 features
    Dropout(0.2),
    x -> x[:, end, :],
    Dense(64 => 32, relu),
    Dropout(0.2),
    Dense(32 => 1)        # Output: next month temp
)

#Train loop
opt = ADAM(0.001)
state = Flux.setup(opt, model)
println("Training...")
for epoch in 1:200
    grads = Flux.gradient(m -> Flux.mse(m(X_train), Y_train), model)
    Flux.update!(state, model, grads[1])
    if epoch % 10 == 0
        train_rmse = sqrt(Flux.mse(model(X_train), Y_train)) * σ
        val_rmse = sqrt(Flux.mse(model(X_val), Y_val)) * σ
        println("Epoch $epoch | Train loss: $(round(train_rmse, digits=2))C | Val loss: $(round(val_rmse, digits=2))C")
    end
end

println("\nDone")