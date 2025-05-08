clc; clear; close all;

%% Step 1: Parameters for OFDM & MIMO
num_subcarriers = 64; % OFDM subcarriers
cp_len = 16; % Cyclic Prefix Length
num_symbols = 1000; % Total symbols to transmit
num_time_instances = num_symbols / 2; % Since it's 2x2 MIMO
NR = 2; NT = 2; % 2x2 MIMO

%% Step 2: Generate Transmitted Symbols (BPSK)
X = randi([0 1], NT, num_subcarriers) * 2 - 1;  % BPSK Mapping {-1, 1}

%% Step 3: OFDM Modulation
X_OFDM = ifft(X, num_subcarriers, 2); % IFFT across subcarriers
X_CP = [X_OFDM(:, end-cp_len+1:end), X_OFDM]; % Add Cyclic Prefix

%% Step 4: MIMO Channel + Noise
H = (randn(NR, NT) + 1j*randn(NR, NT)) / sqrt(2); % Rayleigh Channel
N = 0.2 * (randn(NR, size(X_CP, 2)) + 1j*randn(NR, size(X_CP, 2))); % AWGN Noise
Y_CP = H * X_CP + N; % Received signal

%% Step 5: OFDM Demodulation (Remove CP + FFT)
Y = Y_CP(:, cp_len+1:end); % Remove Cyclic Prefix
Y_OFDM = fft(Y, num_subcarriers, 2); % FFT across subcarriers

%% Step 6: ELM Hidden Layer Processing
L_values = [4, 8, 16, 32, 64]; % Hidden layer sizes
num_pilot_values = 0:50:150; % Number of pilot symbols
accuracy_matrix = zeros(length(L_values), length(num_pilot_values));

for l_idx = 1:length(L_values)
L = L_values(l_idx);
W = randn(L, NR); % Random hidden layer weights
B = randn(L, 1);  % Bias


M = tanh(W * real(Y_OFDM) + B); % Nonlinear transformation


for p_idx = 1:length(num_pilot_values)
num_pilot = min(num_pilot_values(p_idx), num_subcarriers);  % Ensure it does not exceed 64
X_pilot = X(:, 1:num_pilot);  % Select valid pilot symbols
M_pilot = M(:, 1:num_pilot);


% Compute ELM Output Weights (Beta)
alpha = 0.01;  
beta = pinv(M_pilot * M_pilot' + alpha * eye(L)) * M_pilot * X_pilot.';

% Predict Symbols
X_pred = sign(real(beta' * M));  
X_true = X;  

% Compute Accuracy
accuracy = sum(X_pred(:) == X_true(:)) / numel(X_true);
accuracy_matrix(l_idx, p_idx) = accuracy * 100;  


end

end

%% Step 7: Plot Accuracy vs. Number of Pilots
figure;
hold on;
for i = 1:length(L_values)
plot(num_pilot_values, accuracy_matrix(i, :), '-o', 'LineWidth', 2, 'DisplayName', ['L = ' num2str(L_values(i))]);
end
xlabel('Number of Pilot Symbols');
ylabel('ELM Classification Accuracy (%)');
title('ELM Accuracy vs. Number of Pilots');
legend show;
grid on;
hold off;

%% Step 8: Visualize Weight Matrices
figure;
subplot(1, 2, 1);
imagesc(W);
colorbar;
title('Hidden Layer Weights (W)');

subplot(1, 2, 2);
imagesc(beta);
colorbar;
title('Output Weights (Beta)');

%% Step 9: Print Sample Symbols
fprintf('\n=== Sample Predictions vs. Ground Truth ===\n');
disp('X\_true:'); disp(X(:, 1:5));  % First 5 ground truth symbols
disp('X\_pred:'); disp(X_pred(:, 1:5));  
