% Simulation parameters
NT = 2;              % Number of transmit antennas
NR = 2;              % Number of receive antennas
N_FFT = 64;          % FFT points
N_sub = 52;          % Number of sub-carriers
L = 4;               % Number of hidden nodes for ELM
modulation = 'BPSK'; % Modulation type
activation = 'sigmoid'; % Activation function
sampling_freq = 5e6; % Sampling frequency (5 MHz)
alpha = 0.01;        % Regularization parameter for ELM
EbN0_dB_range = 0:5:35; % Eb/N0 range from 0 to 35 dB
num_blocks = 10000;   % Number of blocks for averaging

% Activation function definition
sigmoid = @(x) 1 ./ (1 + exp(-x));

% Possible transmitted symbol vectors for 2x2 BPSK (4 combinations)
X_P = [1, 1; 1, -1; -1, 1; -1, -1].'; % Shape: (2, 4)

% Simulation setup
num_data_symbols = 8; % Data symbols per block
BER_MMSE = zeros(length(EbN0_dB_range), 1); % BER storage for MMSE
BER_ELM = zeros(length(EbN0_dB_range), 1);  % BER storage for ELM

% Initialize ELM parameters (fixed random weights and biases)
W = unifrnd(-1, 1, L, 2 * NR); % L x (2*NR) for real and imaginary parts
B = unifrnd(-1, 1, L, 1);      % L x 1

% Training symbols for channel estimation
X_train1 = [1; 1];   % Shape: (2,1)
X_train2 = [1; -1];  % Shape: (2,1)
X_train = cat(3, X_train1, X_train2); % Shape: (2,1,2)

% Main simulation loop over Eb/N0 values
for idx = 1:length(EbN0_dB_range)
    EbN0_dB = EbN0_dB_range(idx);
    sigma2 = 10 ^ (-EbN0_dB / 10); % Noise variance: Eb/N0 = 1/sigma^2 for BPSK
    sigma = sqrt(sigma2 / 2);      % Per real/imag component
    
    error_MMSE = 0; % Error counter for MMSE
    error_ELM = 0;  % Error counter for ELM
    total_bits = 0; % Total bits transmitted
    
    % Loop over blocks for averaging
    for block = 1:num_blocks
        % Generate channel for this block (constant over block)
        H = (randn(N_sub, NR, NT) + 1i * randn(N_sub, NR, NT)) / sqrt(2);
        
        % Training phase: Estimate channel H using Least Squares (LS)
        X_train_block = repmat(X_train, [1, N_sub, 1, 1]); % Shape: (2,52,1,2)
        X_train_block = permute(X_train_block, [1,2,4,3]); % Shape: (2,52,2,1)
        X_train_block = squeeze(X_train_block); % Shape: (2,52,2)
        N_train = (randn(N_sub, NR, 2) + 1i * randn(N_sub, NR, 2)) * sigma;
        Y_train = zeros(N_sub, NR, 2);
        for sub = 1:N_sub
            Y_train(sub,:,:) = squeeze(H(sub,:,:)) * squeeze(X_train_block(:,sub,:)) + squeeze(N_train(sub,:,:));
        end
        
        % Channel estimation
        H_est = zeros(N_sub, NR, NT);
        for sub = 1:N_sub
            H_est(sub,:,:) = squeeze(Y_train(sub,:,:)) / squeeze(X_train_block(:,sub,:));
        end
        
        % Data transmission
        X_data = sign(randn(NT, N_sub, num_data_symbols)); % BPSK: +1 or -1
        N_data = (randn(N_sub, NR, num_data_symbols) + 1i * randn(N_sub, NR, num_data_symbols)) * sigma;
        Y_data = zeros(N_sub, NR, num_data_symbols);
        for sub = 1:N_sub
            Y_data(sub,:,:) = squeeze(H(sub,:,:)) * squeeze(X_data(:,sub,:)) + squeeze(N_data(sub,:,:));
        end
        
        % Detection per sub-carrier
        for sub = 1:N_sub
            H_sub = squeeze(H_est(sub,:,:)); % [NR, NT], e.g., [2,2]
            
            % MMSE Detection
            for sym = 1:num_data_symbols
                Y = squeeze(Y_data(sub,:,sym)).'; % [NR,1], e.g., [2,1]
                G_MMSE = inv(H_sub' * H_sub + sigma2 * eye(NT)) * H_sub'; % [NT,NR], e.g., [2,2]
                X_hat = G_MMSE * Y; % [NT,NR] * [NR,1] = [NT,1], e.g., [2,1]
                bits_hat = sign(real(X_hat));
                error_MMSE = error_MMSE + sum(bits_hat ~= X_data(:,sub,sym));
            end
            
            % ELM Detection (Proposed Method)
            % Compute Y_P for all possible X_P using estimated channel
            Y_P = zeros(4, NR);
            for k = 1:4
                Y_P(k,:) = H_sub * X_P(:,k);
            end
            
            % Compute M_P for each possible symbol
            M_P = zeros(4, L);
            for k = 1:4
                input_k = [real(Y_P(k,:)), imag(Y_P(k,:))]; % [1, 2*NR]
                input_k = input_k(:).'; % Ensure row vector [1, 2*NR]
                M_P(k,:) = sigmoid(W * input_k' + B); % [L, 2*NR] * [2*NR, 1] = [L, 1]
            end
            
            % Compute β(k) for each possible symbol
            beta_k = zeros(4, NT, L);
            for k = 1:4
                M_k = M_P(k,:); % [1, L]
                norm_M_k = sum(M_k .^ 2) + alpha; % Scalar
                beta_k(k,:,:) = X_P(:,k) * M_k / norm_M_k; % [NT, 1] * [1, L] = [NT, L]
            end
            
            % Detect data symbols using ELM
            for sym = 1:num_data_symbols
                Y = squeeze(Y_data(sub,:,sym)).'; % [NR,1], e.g., [2,1]
                input_n = [real(Y), imag(Y)];
                input_n = input_n(:).'; % Ensure row vector [1, 2*NR]
                M = sigmoid(W * input_n' + B); % [L, 2*NR] * [2*NR, 1] = [L, 1]
                distances = zeros(4,1);
                for k = 1:4
                    distances(k) = norm(Y - Y_P(k,:).');
                end
                [~, k_min] = min(distances);
                X_hat = squeeze(beta_k(k_min,:,:)) * M; % [NT, L] * [L, 1] = [NT, 1]
                bits_hat = sign(real(X_hat));
                error_ELM = error_ELM + sum(bits_hat ~= X_data(:,sub,sym));
            end
        end
        total_bits = total_bits + NT * N_sub * num_data_symbols;
    end
    BER_MMSE(idx) = error_MMSE / total_bits;
    BER_ELM(idx) = error_ELM / total_bits;
end

% Plotting results
figure;
semilogy(EbN0_dB_range, BER_MMSE, 'o-', 'DisplayName', 'MMSE Equalizer');
hold on;
semilogy(EbN0_dB_range, BER_ELM, 's-', 'DisplayName', 'ELM Equalizer');
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER)');
title('BER vs Eb/N0 for 2x2 MIMO-OFDM System');
grid on;
legend;
saveas(gcf, 'BER_vs_EbN0.png');
