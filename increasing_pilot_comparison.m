% Simulation parameters
NT = 2;              % Number of transmit antennas
NR = 2;              % Number of receive antennas
N_FFT = 64;          % FFT points
N_sub = 52;          % Number of sub-carriers
L = 4;               % Number of hidden nodes for ELM
modulation = 'BPSK'; % Modulation type
activation = 'sigmoid'; % Activation function
sampling_freq = 5e6; % Sampling frequency (5 MHz)
doppler_shift = 50;  % Doppler shift (50 Hz)
alpha = 0.01;        % Regularization parameter for ELM
EbN0_dB_range = 0:5:25; % Eb/N0 range from 0 to 25 dB
num_blocks = 1000;   % Number of blocks for averaging

% Activation function definition
sigmoid = @(x) 1 ./ (1 + exp(-x));

% Possible transmitted symbol vectors for 2x2 BPSK (4 combinations)
X_P = [1, 1; 1, -1; -1, 1; -1, -1].'; % Shape: (2, 4)

% Simulation setup
num_data_symbols = 8; % Data symbols per block
num_training_cases = [2, 4, 8]; % Number of training symbols to test
BER_ELM_cases = zeros(length(EbN0_dB_range), length(num_training_cases)); % BER storage for ELM cases

% Initialize ELM parameters (fixed random weights and biases)
W = unifrnd(-1, 1, L, 2 * NR); % L x (2*NR) for real and imaginary parts
B = unifrnd(-1, 1, L, 1);      % L x 1

% Loop over different numbers of training symbols
for case_idx = 1:length(num_training_cases)
    num_train_symbols = num_training_cases(case_idx);
    
    % Define training symbols based on the number of training symbols
    if num_train_symbols == 2
        X_train1 = [1; 1];
        X_train2 = [1; -1];
        X_train = cat(3, X_train1, X_train2); % Shape: (2,1,2)
    elseif num_train_symbols == 4
        X_train1 = [1; 1];
        X_train2 = [1; -1];
        X_train3 = [-1; 1];
        X_train4 = [-1; -1];
        X_train = cat(3, X_train1, X_train2, X_train3, X_train4); % Shape: (2,1,4)
    else % num_train_symbols == 8
        X_train1 = [1; 1];
        X_train2 = [1; -1];
        X_train3 = [-1; 1];
        X_train4 = [-1; -1];
        % Repeat the pattern to get 8 unique combinations (or as orthogonal as possible)
        X_train5 = [1; 1];
        X_train6 = [1; -1];
        X_train7 = [-1; 1];
        X_train8 = [-1; -1];
        X_train = cat(3, X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8); % Shape: (2,1,8)
    end

    % Main simulation loop over Eb/N0 values
    for idx = 1:length(EbN0_dB_range)
        EbN0_dB = EbN0_dB_range(idx);
        sigma2 = 10 ^ (-EbN0_dB / 10); % Noise variance: Eb/N0 = 1/sigma^2 for BPSK
        sigma = sqrt(sigma2 / 2);      % Per real/imag component
        
        error_ELM = 0;  % Error counter for ELM
        total_bits = 0; % Total bits transmitted
        
        % Loop over blocks for averaging
        for block = 1:num_blocks
            % Generate channel for this block (constant over block)
            H = (randn(N_sub, NR, NT) + 1i * randn(N_sub, NR, NT)) / sqrt(2);
            
            % Training phase: Estimate channel H using Least Squares (LS)
            X_train_block = repmat(X_train, [1, N_sub, 1, 1]); % Shape: (2,52,1,num_train_symbols)
            X_train_block = permute(X_train_block, [1,2,4,3]); % Shape: (2,52,num_train_symbols,1)
            X_train_block = squeeze(X_train_block); % Shape: (2,52,num_train_symbols)
            N_train = (randn(N_sub, NR, num_train_symbols) + 1i * randn(N_sub, NR, num_train_symbols)) * sigma;
            Y_train = zeros(N_sub, NR, num_train_symbols);
            for sub = 1:N_sub
                Y_train(sub,:,:) = squeeze(H(sub,:,:)) * squeeze(X_train_block(:,sub,:)) + squeeze(N_train(sub,:,:));
            end
            
            % Channel estimation
            H_est = zeros(N_sub, NR, NT);
            for sub = 1:N_sub
                % Least Squares estimation: H_est = Y_train * pinv(X_train_block)
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
                
                % ELM Detection (Proposed Method)
                Y_P = zeros(4, NR);
                for k = 1:4
                    Y_P(k,:) = H_sub * X_P(:,k);
                end
                
                M_P = zeros(4, L);
                for k = 1:4
                    input_k = [real(Y_P(k,:)), imag(Y_P(k,:))];
                    input_k = input_k(:).'; % Ensure row vector [1, 2*NR]
                    M_P(k,:) = sigmoid(W * input_k' + B);
                end
                
                beta_k = zeros(4, NT, L);
                for k = 1:4
                    M_k = M_P(k,:);
                    norm_M_k = sum(M_k .^ 2) + alpha;
                    beta_k(k,:,:) = X_P(:,k) * M_k / norm_M_k;
                end
                
                for sym = 1:num_data_symbols
                    Y = squeeze(Y_data(sub,:,sym)).';
                    input_n = [real(Y), imag(Y)];
                    input_n = input_n(:).';
                    M = sigmoid(W * input_n' + B);
                    distances = zeros(4,1);
                    for k = 1:4
                        distances(k) = norm(Y - Y_P(k,:).');
                    end
                    [~, k_min] = min(distances);
                    X_hat = squeeze(beta_k(k_min,:,:)) * M;
                    bits_hat = sign(real(X_hat));
                    error_ELM = error_ELM + sum(bits_hat ~= X_data(:,sub,sym));
                end
            end
            total_bits = total_bits + NT * N_sub * num_data_symbols;
        end
        BER_ELM_cases(idx, case_idx) = error_ELM / total_bits;
    end
end

% Plotting results
figure;
semilogy(EbN0_dB_range, BER_ELM_cases(:,1), 's-', 'DisplayName', 'ELM (2 Training Symbols)');
hold on;
semilogy(EbN0_dB_range, BER_ELM_cases(:,2), 'o-', 'DisplayName', 'ELM (4 Training Symbols)');
semilogy(EbN0_dB_range, BER_ELM_cases(:,3), '^-', 'DisplayName', 'ELM (8 Training Symbols)');
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER)');
title('BER vs Eb/N0 for ELM Receiver with Different Numbers of Training Symbols');
grid on;
legend;
saveas(gcf, 'BER_vs_EbN0_Training_Symbols_Comparison.png');
