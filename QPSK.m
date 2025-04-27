% Simulation parameters
NT = 4;              % Number of transmit antennas
NR = 4;              % Number of receive antennas
N_FFT = 64;          % FFT points
N_sub = 48;          % Number of sub-carriers
L = 16;              % Number of hidden nodes for ELM
modulation = 'QPSK'; % Modulation type
activation = 'sigmoid'; % Activation function
sampling_freq = 5e6; % Sampling frequency (5 MHz)
alpha = 0.01;        % Regularization parameter for ELM
EbN0_dB_range = 0:5:25; % Extended Eb/N0 range
num_blocks = 20000;   % Number of blocks for averaging

% Activation function definition
sigmoid = @(x) 1 ./ (1 + exp(-x));

% Possible transmitted symbol vectors for 4x4 QPSK (4^4 = 256 combinations)
QPSK_symbols = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j] / sqrt(2); % Normalized QPSK symbols
X_P = zeros(NT, 256);
idx = 1;
for i = 1:4
    for j = 1:4
        for k = 1:4
            for m = 1:4
                X_P(:, idx) = [QPSK_symbols(i); QPSK_symbols(j); QPSK_symbols(k); QPSK_symbols(m)];
                idx = idx + 1;
            end
        end
    end
end

% Simulation setup
num_data_symbols = 8; % Data symbols per block
BER_MMSE = zeros(length(EbN0_dB_range), 1); % BER storage for MMSE
BER_ELM = zeros(length(EbN0_dB_range), 1);  % BER storage for ELM

% Initialize ELM parameters
W = unifrnd(-1, 1, L, 2 * NR); % L x (2*NR)
B = unifrnd(-1, 1, L, 1);      % L x 1

% Training symbols for channel estimation (minimum 4 for QPSK, orthogonal using Hadamard)
hadamard_matrix = hadamard(4); % 4x4 Hadamard matrix
X_train = zeros(NT, 1, 4);
for t = 1:4
    X_train(:, :, t) = (hadamard_matrix(t, :) * [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]') / sqrt(2);
end

% Main simulation loop over Eb/N0 values
for idx = 1:length(EbN0_dB_range)
    EbN0_dB = EbN0_dB_range(idx);
    sigma2 = 10 ^ (-EbN0_dB / 10) / 2; % Normalized for 2 bits/symbol
    sigma = sqrt(sigma2 / 2);          % Per real/imag component
    
    error_MMSE = 0; % Error counter for MMSE
    error_ELM = 0;  % Error counter for ELM
    total_bits = 0; % Total bits transmitted
    
    for block = 1:num_blocks
        % Generate channel
        H = (randn(N_sub, NR, NT) + 1i * randn(N_sub, NR, NT)) / sqrt(2);
        
        % Training phase: Estimate channel H using LS with regularization
        X_train_block = repmat(X_train, [1, N_sub, 1, 1]); % Shape: (4,48,1,4)
        X_train_block = permute(X_train_block, [1,2,4,3]); % Shape: (4,48,4,1)
        X_train_block = squeeze(X_train_block); % Shape: (4,48,4)
        N_train = (randn(N_sub, NR, 4) + 1i * randn(N_sub, NR, 4)) * sigma;
        Y_train = zeros(N_sub, NR, 4);
        for sub = 1:N_sub
            Y_train(sub,:,:) = squeeze(H(sub,:,:)) * squeeze(X_train_block(:,sub,:)) + squeeze(N_train(sub,:,:));
        end
        
        % LS estimation with regularization
        H_est = zeros(N_sub, NR, NT);
        reg_lambda = 1e-6; % Small regularization term
        for sub = 1:N_sub
            X_mat = squeeze(X_train_block(:,sub,:)); % (4, NT)
            Y_mat = squeeze(Y_train(sub,:,:));       % (NR, 4)
            H_est(sub,:,:) = (Y_mat * X_mat') * pinv(X_mat * X_mat' + reg_lambda * eye(NT)); % Regularized LS
        end
        
        % Data transmission
        X_data_real = sign(randn(NT, N_sub, num_data_symbols));
        X_data_imag = sign(randn(NT, N_sub, num_data_symbols));
        X_data = (X_data_real + 1j * X_data_imag) / sqrt(2);
        N_data = (randn(N_sub, NR, num_data_symbols) + 1i * randn(N_sub, NR, num_data_symbols)) * sigma;
        Y_data = zeros(N_sub, NR, num_data_symbols);
        for sub = 1:N_sub
            Y_data(sub,:,:) = squeeze(H(sub,:,:)) * squeeze(X_data(:,sub,:)) + squeeze(N_data(sub,:,:));
        end
        
        % Detection per sub-carrier
        for sub = 1:N_sub
            H_sub = squeeze(H_est(sub,:,:)); % [NR, NT], e.g., [4,4]
            
            % MMSE Detection with pinv
            for sym = 1:num_data_symbols
                Y = squeeze(Y_data(sub,:,sym)).'; % [NR,1], e.g., [4,1]
                G_MMSE = pinv(H_sub' * H_sub + sigma2 * eye(NT)) * H_sub'; % Use pinv for robustness
                X_hat = G_MMSE * Y; % [NT,1]
                bits_hat_real = sign(real(X_hat));
                bits_hat_imag = sign(imag(X_hat));
                bits_hat = [bits_hat_real; bits_hat_imag]; % [NT*2,1]
                X_data_bits = [sign(real(X_data(:,sub,sym))); sign(imag(X_data(:,sub,sym)))];
                error_MMSE = error_MMSE + sum(bits_hat ~= X_data_bits);
            end
            
            % ELM Detection
            Y_P = zeros(256, NR);
            for k = 1:256
                Y_P(k,:) = H_sub * X_P(:,k);
            end
            
            M_P = zeros(256, L);
            for k = 1:256
                input_k = [real(Y_P(k,:)), imag(Y_P(k,:))];
                input_k = input_k(:).';
                M_P(k,:) = sigmoid(W * input_k' + B);
            end
            
            beta_k = zeros(256, NT, L);
            for k = 1:256
                M_k = M_P(k,:);
                norm_M_k = sum(M_k .^ 2) + alpha;
                beta_k(k,:,:) = X_P(:,k) * M_k / norm_M_k;
            end
            
            for sym = 1:num_data_symbols
                Y = squeeze(Y_data(sub,:,sym)).';
                input_n = [real(Y), imag(Y)];
                input_n = input_n(:).';
                M = sigmoid(W * input_n' + B);
                distances = zeros(256,1);
                for k = 1:256
                    distances(k) = norm(Y - Y_P(k,:).');
                end
                [~, k_min] = min(distances);
                X_hat = squeeze(beta_k(k_min,:,:)) * M;
                bits_hat_real = sign(real(X_hat));
                bits_hat_imag = sign(imag(X_hat));
                bits_hat = [bits_hat_real; bits_hat_imag];
                X_data_bits = [sign(real(X_data(:,sub,sym))); sign(imag(X_data(:,sub,sym)))];
                error_ELM = error_ELM + sum(bits_hat ~= X_data_bits);
            end
        end
        total_bits = total_bits + 2 * NT * N_sub * num_data_symbols; % 2 bits per QPSK symbol
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
title('BER vs Eb/N0 for 4x4 MIMO-OFDM System with QPSK');
grid on;
legend;
saveas(gcf, 'BER_vs_EbN0_QPSK_4x4.png');
