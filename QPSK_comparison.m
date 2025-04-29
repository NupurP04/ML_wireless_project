clc;
clear;
close all;

%% Parameters
Nt = 2;                          % Number of transmit antennas
Nr = 2;                          % Number of receive antennas
N_fft = 64;                      % FFT size
cp_len = 2;                      % Cyclic prefix length

% Modulation settings (toggle between QPSK and 16-QAM)
modType = 'QPSK';                % Options: 'QPSK' or '16QAM'
if strcmp(modType, 'QPSK')
    modOrder = 4;                % QPSK modulation order
    bitsPerSymbol = 2;           % Bits per symbol for QPSK
    num_pilots = 4;              % Minimal pilots: 2^2 = 4
else % 16-QAM
    modOrder = 16;               % 16-QAM modulation order
    bitsPerSymbol = 4;           % Bits per symbol for 16-QAM
    num_pilots = 16;             % Minimal pilots: 2^4 = 16
end

num_symbols = N_fft - num_pilots - 1; % Number of data symbols
numBits = 50 * Nt * num_symbols * bitsPerSymbol; % Total bits
fs = 5e6;                        % Sampling frequency (Hz)
fd = 50;                         % Doppler shift (Hz)
Pt = 1;                          % Transmit power
time_stamps = numBits / (Nt * num_symbols * bitsPerSymbol); % Number of OFDM symbols
SNR_dB = 0:5:30;                 % SNR range (dB)
BER_MMSE = zeros(size(SNR_dB));  % BER for MMSE
BER_ELM = zeros(size(SNR_dB));   % BER for ELM

% ELM Parameters
L = 4;                           % Number of hidden nodes
alpha = 0.01;                    % Regularization parameter
activation = @(x) 1./(1+exp(-x)); % Sigmoid activation function
W = unifrnd(-1,1,L,2*Nr);        % Random weights
B = unifrnd(-1,1,L,1);           % Random biases

% All possible MIMO symbol vectors
if strcmp(modType, 'QPSK')
    qpsk_symbols = [1+1j, 1-1j, -1+1j, -1-1j] / sqrt(2); % Normalized QPSK
    num_possible = modOrder^Nt;   % 4^2 = 16 for 2x2 MIMO
    X_P = zeros(Nt, num_possible);
    idx = 1;
    for i = 1:4
        for j = 1:4
            X_P(:, idx) = [qpsk_symbols(i); qpsk_symbols(j)];
            idx = idx + 1;
        end
    end
else % 16-QAM
    qam_symbols = qammod(0:15, 16, 'UnitAveragePower', true); % 16-QAM symbols
    num_possible = modOrder^Nt;   % 16^2 = 256 (use subset for simplicity)
    X_P = zeros(Nt, 16);          % Use 16 vectors for practicality
    for idx = 1:16
        X_P(:, idx) = [qam_symbols(idx); qam_symbols(mod(idx+1,16)+1)];
    end
end

%% Generate random bits
bits = randi([0 1], numBits, 1);
symbols = qammod(bits, modOrder, 'InputType', 'bit', 'UnitAveragePower', true);
symbols = reshape(symbols, Nt, num_symbols, time_stamps);

%% MIMO Channel Object
rayleighChan = comm.MIMOChannel( ...
    'FadingDistribution', 'Rayleigh', ...
    'SampleRate', fs, ...
    'MaximumDopplerShift', fd, ...
    'NumTransmitAntennas', Nt, ...
    'NumReceiveAntennas', Nr, ...
    'SpatialCorrelationSpecification', 'None');

%% Loop over SNR values
for snr_idx = 1:length(SNR_dB)
    snr_linear = 10^(SNR_dB(snr_idx)/10);
    noise_variance = Pt / snr_linear;  
    noise_std = sqrt(noise_variance/2);

    % --- Transmitter Side ---
    ofdm_symbols = zeros(N_fft, Nt, time_stamps);
    ofdm_no_pilot_symbols = pagetranspose(symbols);
    
    % Insert pilots (minimal number based on modulation)
    if strcmp(modType, 'QPSK')
        pilot_symbols = qpsk_symbols; % 4 pilots
        for p = 1:num_pilots
            ofdm_symbols(p,1,:) = pilot_symbols(p);
            ofdm_symbols(p,2,:) = pilot_symbols(mod(p,4)+1); % Orthogonal pilots
        end
    else % 16-QAM
        pilot_symbols = qam_symbols(1:num_pilots); % 16 pilots
        for p = 1:num_pilots
            ofdm_symbols(p,1,:) = pilot_symbols(p);
            ofdm_symbols(p,2,:) = pilot_symbols(mod(p-1,16)+1); % Distinct pilots
        end
    end

    % Insert data
    k = (N_fft/2) - num_pilots;
    for i = 1:time_stamps
        ofdm_symbols(num_pilots+1:N_fft/2,:,i) = ofdm_no_pilot_symbols(1:k,:,i);
        ofdm_symbols(N_fft/2+2:end,:,i) = ofdm_no_pilot_symbols(k+1:end,:,i);
    end

    % IFFT
    ofdm_symbols_time = ifft(ofdm_symbols, N_fft);

    % Add CP
    ofdm_symbols_cp = cat(1, ofdm_symbols_time(end-cp_len+1:end,:,:), ofdm_symbols_time);
    txFrame = sqrt(Pt) * ofdm_symbols_cp;

    detected_MMSE = zeros(size(ofdm_no_pilot_symbols)); % Nt x num_symbols x time_stamps
    detected_ELM = zeros(size(ofdm_no_pilot_symbols));

    for i = 1:time_stamps
        % --- Channel ---
        H_actual = (randn(Nr,Nt)+1j*randn(Nr,Nt))/sqrt(2); 
        rxFrame = zeros(size(txFrame(:,:,i)));
        for n = 1:N_fft+cp_len
            rxFrame(n,:) = rayleighChan(txFrame(n,:,i));
        end

        % --- Add Noise ---
        noise = noise_std*(randn(size(rxFrame)) + 1i*randn(size(rxFrame)));
        rxFrame = rxFrame + noise;

        % --- Receiver ---
        rxFrame_no_cp = rxFrame(cp_len+1:end, :);
        rxFrame_freq = fft(rxFrame_no_cp, N_fft);

        % --- Channel Estimation using minimal pilots ---
        Y_pilot = rxFrame_freq(1:num_pilots,:).';       % Nr x num_pilots
        X_pilot = squeeze(ofdm_symbols(1:num_pilots,:,i)).'; % Nt x num_pilots
        H_est = Y_pilot * pinv(X_pilot);                % Nr x Nt (2 x 2)

        % --- Data detection ---
        data_idx = [num_pilots+1:N_fft/2, N_fft/2+2:N_fft];
        rx_data = rxFrame_freq(data_idx,:);

        for h = 1:length(data_idx)
            Y = rx_data(h,:).'; % Nr x 1

            % --- MMSE Detection ---
            G_MMSE = (H_est'*H_est + noise_variance*eye(Nt)) \ (H_est');
            X_hat_MMSE = G_MMSE * Y; % Nt x 1
            detected_MMSE(h,:,i) = X_hat_MMSE; % Assign complex symbols

            % --- ELM Detection ---
            num_vec = size(X_P, 2); % 16 for QPSK, 16 for 16-QAM subset
            Y_P = zeros(num_vec, Nr);
            for k_p = 1:num_vec
                Y_P(k_p,:) = (H_est) * X_P(:,k_p);
            end

            M_P = zeros(num_vec, L);
            for k_p = 1:num_vec
                input_k = [real(Y_P(k_p,:)), imag(Y_P(k_p,:))];
                M_P(k_p,:) = activation(W * input_k' + B).';
            end

            beta_k = zeros(num_vec, Nt, L);
            for k_p = 1:num_vec
                norm_Mk = sum(M_P(k_p,:).^2) + alpha;
                beta_k(k_p,:,:) = X_P(:,k_p) * M_P(k_p,:) / norm_Mk;
            end

            input_n = [real(Y.'), imag(Y.')];
            M_n = activation(W * input_n' + B);

            distances = zeros(num_vec, 1);
            for k_p = 1:num_vec
                distances(k_p) = norm(Y - Y_P(k_p,:).');
            end
            [~,k_min] = min(distances);

            X_hat_ELM = squeeze(beta_k(k_min,:,:)) * M_n;
            detected_ELM(h,:,i) = X_hat_ELM; % Assign complex symbols
        end
    end

    % BER Computation
    transmitted_symbols = ofdm_no_pilot_symbols(:); % Vectorize for comparison
    detected_MMSE_symbols = detected_MMSE(:);
    detected_ELM_symbols = detected_ELM(:);

    % Convert to bits for BER
    transmitted_bits = qamdemod(transmitted_symbols, modOrder, 'OutputType', 'bit');
    detected_MMSE_bits = qamdemod(detected_MMSE_symbols, modOrder, 'OutputType', 'bit');
    detected_ELM_bits = qamdemod(detected_ELM_symbols, modOrder, 'OutputType', 'bit');

    errors_MMSE = sum(transmitted_bits ~= detected_MMSE_bits);
    errors_ELM = sum(transmitted_bits ~= detected_ELM_bits);

    BER_MMSE(snr_idx) = errors_MMSE / numBits;
    BER_ELM(snr_idx) = errors_ELM / numBits;
end

%% Plot
figure;
semilogy(SNR_dB, BER_MMSE, 'bo-','LineWidth',2,'MarkerFaceColor','b','DisplayName','MMSE');
hold on;
semilogy(SNR_dB, BER_ELM, 'rs-','LineWidth',2,'MarkerFaceColor','r','DisplayName','ELM');
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title(['BER vs SNR for 2x2 MIMO-OFDM (', modType, ')']);
legend;
ylim([1e-4 1]);
xlim([0 35]);
