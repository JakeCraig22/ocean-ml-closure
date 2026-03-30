using Flux, HTTP, JSON3, Dates, Statistics, Random


LAT, LON = 43.6591, -70.2568      # Portland, ME
SEQ_LEN, PRED_DAYS = 30, 7        #  30 day → 7 day


end_date = today() - Day(1)
start_date = end_date - Year(3)  # 3 years of daily data

# use archive API for historical data (daily average temperature)
url = "https://archive-api.open-meteo.com/v1/archive?latitude=$LAT&longitude=$LON&daily=temperature_2m_max,temperature_2m_min&start_date=$start_date&end_date=$end_date&timezone=auto"
println("Fetching data from: $url")

temp, humidity, wind_speed, pressure = nothing, nothing, nothing, nothing
dates = nothing

try
    response = HTTP.get(url)
    data = JSON3.read(String(response.body))

    # archive API: daily max/min temperature
    daily_temp_max = data.daily.temperature_2m_max
    daily_temp_min = data.daily.temperature_2m_min

    # filter null and convert to Float32
    valid_max = [Float32(t) for t in daily_temp_max if t !== nothing]
    valid_min = [Float32(t) for t in daily_temp_min if t !== nothing]

    if length(valid_max) > 0 && length(valid_min) > 0
        # average of max and min as daily mean temperature
        n_days = min(length(valid_max), length(valid_min))
        global temp = [(valid_max[i] + valid_min[i]) / 2 for i in 1:n_days]
        global dates = [start_date + Day(i-1) for i in 1:n_days]
        # use same value for other variables (placeholder)
        global humidity = fill(50.0f0, n_days)
        global wind_speed = fill(5.0f0, n_days)
        global pressure = fill(1013.0f0, n_days)
        println("data: $(length(temp)) days | from: $(dates[1]) to $(dates[end])")
        println("variables: temp (daily mean from max/min), humidity, wind_speed, pressure (placeholder)")
        

        nan_count = count(isnan, temp)
        if nan_count > 0
            println("WARNING: API returned $nan_count NaN values in temperature data!")
            nan_indices = findall(isnan, temp)
            println("NaN positions: $(nan_indices[1:min(10, length(nan_indices))])...")
        else
            println("Data check: No NaN values in API response")
        end
    end
catch e
    println("API fail: $e")
end

if temp === nothing
    println("no data, exit.")
    exit()
end

nan_mask = isnan.(temp)
if any(nan_mask)
    nan_count = sum(nan_mask)
    println("Found $nan_count NaN values in temperature data. Removing affected days...")
    valid_mask = .!nan_mask
    global temp = temp[valid_mask]
    global humidity = humidity[valid_mask]
    global wind_speed = wind_speed[valid_mask]
    global pressure = pressure[valid_mask]
    global dates = dates[valid_mask]
    println("Cleaned data: $(length(temp)) days remaining")
end

# check if NaN data
if any(isnan.(temp))
    println("ERROR: Still have NaN in temperature data after cleaning. Cannot train model.")
    exit()
end

σ_temp = std(temp)
if σ_temp == 0 || isnan(σ_temp)
    println("ERROR: Temperature std is 0 or NaN. Cannot normalize data.")
    exit()
end


# Normalize
μ_temp = mean(temp)  # mean already computed, recalculate for clarity
μ_hum, σ_hum = mean(humidity), std(humidity)
μ_wind, σ_wind = mean(wind_speed), std(wind_speed)
μ_pres, σ_pres = mean(pressure), std(pressure)


if σ_hum == 0 || isnan(σ_hum); σ_hum = 1.0f0; end
if σ_wind == 0 || isnan(σ_wind); σ_wind = 1.0f0; end
if σ_pres == 0 || isnan(σ_pres); σ_pres = 1.0f0; end

temp_norm = (temp .- μ_temp) ./ σ_temp
humidity_norm = (humidity .- μ_hum) ./ σ_hum
wind_speed_norm = (wind_speed .- μ_wind) ./ σ_wind
pressure_norm = (pressure .- μ_pres) ./ σ_pres

# input：[temp, humidity, wind_speed, pressure, sin, cos] = 6 features
n_features = 6
n_samples = length(temp_norm) - SEQ_LEN - PRED_DAYS + 1
X = zeros(Float32, n_features, SEQ_LEN, n_samples)
Y = zeros(Float32, PRED_DAYS, n_samples)

for i in 1:n_samples
    X[1, :, i] = temp_norm[i:i+SEQ_LEN-1]
    X[2, :, i] = humidity_norm[i:i+SEQ_LEN-1]
    X[3, :, i] = wind_speed_norm[i:i+SEQ_LEN-1]
    X[4, :, i] = pressure_norm[i:i+SEQ_LEN-1]
    t = Float64.(i:i+SEQ_LEN-1)
    X[5, :, i] = sin.(2π .* t ./ 365.25)
    X[6, :, i] = cos.(2π .* t ./ 365.25)
    Y[:, i] = temp_norm[i+SEQ_LEN : i+SEQ_LEN+PRED_DAYS-1]
end

n_train = floor(Int, 0.8 * n_samples)
X_train, Y_train = X[:, :, 1:n_train], Y[:, 1:n_train]
X_val, Y_val = X[:, :, n_train+1:end], Y[:, n_train+1:end]

# Flux LSTM expects: (input_dim, batch_size, seq_len)
# We need to reshape X from (features, seq_len, batch) to (features, batch, seq_len)
function reshape_for_lstm(x)
    # x: (features, seq_len, batch) -> (features, batch, seq_len)
    return permutedims(x, (1, 3, 2))
end

X_train_lstm = reshape_for_lstm(X_train)
X_val_lstm = reshape_for_lstm(X_val)


model = Chain(
    LSTM(n_features => 128),
    Dropout(0.2),                   
    x -> x[:, :, end],
    Dense(128 => 64, relu),
    Dropout(0.2),
    Dense(64 => 32, relu),
    Dense(32 => PRED_DAYS)
)

# early stopping parameters
PATIENCE = 20
MIN_DELTA = 0.001f0
best_val_loss = Inf32
patience_counter = 0
best_epoch = 0
best_model_params = nothing  # save best model parameters

opt = ADAM(0.001)
state = Flux.setup(opt, model)
println("train...")
println("Early stopping: patience=$PATIENCE, min_delta=$MIN_DELTA")
println("-" ^ 55)

for epoch in 1:200

    if epoch % 50 == 0 && epoch > 0
        new_lr = opt.eta / 2
        opt.eta = new_lr
        println("  Learning rate adjusted to: $new_lr")
    end
    
    grads = Flux.gradient(model) do m
        pred = m(X_train_lstm)
        Flux.mse(pred, Y_train)
    end
    Flux.update!(state, model, grads[1])
    
    if epoch % 10 == 0
        val_pred = model(X_val_lstm)
        val_rmse = sqrt(Flux.mse(val_pred, Y_val)) * σ_temp
        train_rmse = sqrt(Flux.mse(model(X_train_lstm), Y_train)) * σ_temp
        println("Epoch $epoch / 200 | Train RMSE: $(round(train_rmse, digits=2))°C | Val RMSE: $(round(val_rmse, digits=2))°C")
        
        # early stop
        if best_val_loss - val_rmse > MIN_DELTA
            global best_val_loss = val_rmse
            global patience_counter = 0
            global best_epoch = epoch
            # save modle
            global best_model_params = Flux.params(model) .|> deepcopy
            println("  ✓ New best model! (Val RMSE: $(round(val_rmse, digits=2))°C)")
        else
            global patience_counter += 1
            println("  Patience: $patience_counter / $PATIENCE (Best: Epoch $best_epoch, RMSE: $(round(best_val_loss, digits=2))°C)")
        end

        if patience_counter >= PATIENCE
            println("\nEarly stopping triggered at Epoch $epoch!")
            println("Restoring best model from Epoch $best_epoch...")
            Flux.loadmodel!(model, best_model_params)
            break
        end
    end
end

println("-" ^ 55)
println("Training finished. Best Epoch: $best_epoch, Best Val RMSE: $(round(best_val_loss, digits=2))°C")


X_last = zeros(Float32, n_features, SEQ_LEN, 1)
X_last[1, :, 1] = temp_norm[end-SEQ_LEN+1:end]
X_last[2, :, 1] = humidity_norm[end-SEQ_LEN+1:end]
X_last[3, :, 1] = wind_speed_norm[end-SEQ_LEN+1:end]
X_last[4, :, 1] = pressure_norm[end-SEQ_LEN+1:end]
t_last = (n_samples:n_samples+SEQ_LEN-1)
X_last[5, :, 1] = sin.(2π .* t_last ./ 365.25)
X_last[6, :, 1] = cos.(2π .* t_last ./ 365.25)
# Reshape for LSTM: (features, seq_len, batch) -> (features, batch, seq_len)
X_last_lstm = reshape_for_lstm(X_last)
pred_norm = model(X_last_lstm) |> Array
pred_temp = vec(pred_norm) .* σ_temp .+ μ_temp
pred_dates = dates[end] .+ Day.(1:PRED_DAYS)

println("\n Portland, ME - 7 day (°C):")
println("-" ^ 40)
for (i, d) in enumerate(pred_dates)
    println("$(Dates.format(d, "yyyy-mm-dd")) : $(round(pred_temp[i], digits=1))°C")
end
println("-" ^ 40)
println("mean $(round(mean(pred_temp), digits=1))°C | range $(round(minimum(pred_temp), digits=1))~$(round(maximum(pred_temp), digits=1))°C")
