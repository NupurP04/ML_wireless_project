clc;
clear;
close all;

%% Parameters
Nt = 2;                          % Number of transmit antennas
Nr = 2;                          % Number of receive antennas
N_fft = 64;                      % FFT size
cp_len = 2;                      % Cyclic prefix length
modOrder = 2;                    % BPSK modulation order
bitsPerSymbol = log2(modOrder);  % Bits per symbol (1 for BPSK)
num_pilots = 2;                  % Minimal pilots as per paper's novelty
num_symbols = N_fft - num_pilots - 1; % Number of data symbols
numBits = 50 * Nt * num_symbols; % Total bits
fs = 5e6;                        % Sampling frequency (Hz)
Pt = 1;                          % Transmit power
time_stamps = numBits / (Nt * num_symbols); % Number of OFDM symbols
SNR_dB = 0:5:30;                 % SNR range (dB)
doppler_freqs = [5, 50, 500];    % Doppler frequencies (Hz) for low, medium, high mobility

% Preallocate arrays
BER_MMSE = zeros(length(doppler_freqs), length(SNR_dB));
BER_ELM = zeros(length(doppler_freqs), length(SNR_dB));

% ELM Parameters
L = 4;                           % Number of hidden nodes
alpha = 0.01;                    % Regularization parameter
activation = @(x) 1./(1+exp(-x)); % Sigmoid activation function
W = unifrnd(-1,1,L,2*Nr);        % Random weights
B = unifrnd(-1,1,L,1);           % Random biases

% All possible BPSK MIMO symbol vectors (2^N_T = 4 for 2 antennas)
X_P = [1 1; 1 -1; -1 1; -1 -1]';

%% Generate random bits
bits = randi([0 1], numBits, 1);
symbols = 2 * bits - 1;  % BPSK mapping: 0 -> -1, 1 -> 1

%% Loop over Doppler frequencies
for dop_idx = 1:length(doppler_freqs)
    fd = doppler_freqs(dop_idx);
    rayleighChan = comm.MIMOChannel( ...
        'FadingDistribution', 'Rayleigh', ...
        'SampleRate', fs, ...
        'MaximumDopplerShift', fd, ...
        'NumTransmitAntennas', Nt, ...
        'NumReceiveAntennas', Nr, ...
        'SpatialCorrelationSpecification', 'None');

    % --- Transmitter Side ---
    ofdm_symbols = zeros(N_fft, Nt, time_stamps);
    ofdm_no_pilot_symbols = pagetranspose(reshape(symbols, Nt, num_symbols, []));
    
    % Insert orthogonal pilots (minimal 2 pilots)
    ofdm_symbols(1,1,:) = 1;    ofdm_symbols(1,2,:) = 1;    % Pilot subcarrier 1
    ofdm_symbols(2,1,:) = 1;    ofdm_symbols(2,2,:) = -1;   % Pilot subcarrier 2

    % Insert data
    k = (N_fft/2) - num_pilots; % k = 32 - 2 = 30
    for i = 1:time_stamps
        ofdm_symbols(num_pilots+1:N_fft/2,:,i) = ofdm_no_pilot_symbols(1:k,:,i);
        ofdm_symbols(N_fft/2+2:end,:,i) = ofdm_no_pilot_symbols(k+1:end,:,i);
    end

    % IFFT
    ofdm_symbols_time = ifft(ofdm_symbols, N_fft);

    % Add CP
    ofdm_symbols_cp = cat(1, ofdm_symbols_time(end-cp_len+1:end,:,:), ofdm_symbols_time);
    txFrame = sqrt(Pt) * ofdm_symbols_cp;

    detected_MMSE = zeros(size(ofdm_no_pilot_symbols));
    detected_ELM = zeros(size(ofdm_no_pilot_symbols));

    %% Loop over SNR values
    for snr_idx = 1:length(SNR_dB)
        snr_linear = 10^(SNR_dB(snr_idx)/10);
        noise_variance = Pt / snr_linear;  
        noise_std = sqrt(noise_variance/2);

        for i = 1:time_stamps
            % --- Channel ---
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
            Y_pilot = rxFrame_freq(1:num_pilots,:).';       % Nr x num_pilots (2 x 2)
            X_pilot = squeeze(ofdm_symbols(1:num_pilots,:,i)).'; % Nt x num_pilots (2 x 2)
            H_est = Y_pilot * pinv(X_pilot);                % Nr x Nt (2 x 2)

            % --- Data detection ---
            data_idx = [num_pilots+1:N_fft/2, N_fft/2+2:N_fft]; % [3:32, 34:64]
            rx_data = rxFrame_freq(data_idx,:);

            for h = 1:length(data_idx)
                Y = rx_data(h,:).'; % Nr x 1

                % --- MMSE Detection ---
                G_MMSE = (H_est'*H_est + noise_variance*eye(Nt)) \ (H_est');
                X_hat_MMSE = G_MMSE * Y;
                detected_MMSE(h,:,i) = sign(real(X_hat_MMSE));

                % --- ELM Detection ---
                % Compute Y_P for all possible symbol vectors
                Y_P = zeros(4,Nr);
                for k_p = 1:4
                    Y_P(k_p,:) = (H_est) * X_P(:,k_p);
                end

                % Compute M_P for each possible vector
                M_P = zeros(4,L);
                for k_p = 1:4
                    input_k = [real(Y_P(k_p,:)), imag(Y_P(k_p,:))];
                    input_k = input_k(:).';
                    M_P(k_p,:) = activation(W * input_k' + B).';
                end

                % Compute beta_k for each possible vector
                beta_k = zeros(4,Nt,L);
                for k_p = 1:4
                    norm_Mk = sum(M_P(k_p,:).^2) + alpha;
                    beta_k(k_p,:,:) = X_P(:,k_p) * M_P(k_p,:) / norm_Mk;
                end

                % Hidden layer output for received signal
                input_n = [real(Y.'), imag(Y.')];
                input_n = input_n(:).';
                M_n = activation(W * input_n' + B);

                % Minimum distance detection
                distances = zeros(4,1);
                for k_p = 1:4
                    distances(k_p) = norm(Y - Y_P(k_p,:).');
                end
                [~,k_min] = min(distances);

                X_hat_ELM = squeeze(beta_k(k_min,:,:)) * M_n;
                detected_ELM(h,:,i) = sign(real(X_hat_ELM));
            end
        end

        % BER Computation
        transmitted_data = ofdm_no_pilot_symbols;
        errors_MMSE = sum(transmitted_data(:) ~= detected_MMSE(:));
        errors_ELM = sum(transmitted_data(:) ~= detected_ELM(:));

        BER_MMSE(dop_idx, snr_idx) = errors_MMSE/numBits;
        BER_ELM(dop_idx, snr_idx) = errors_ELM/numBits;
    end
end

%% Plot
% figure;
% for dop_idx = 1:length(doppler_freqs)
%     subplot(length(doppler_freqs), 1, dop_idx);
%     semilogy(SNR_dB, BER_MMSE(dop_idx, :), 'bo-', 'LineWidth', 2, 'DisplayName', 'MMSE');
%     hold on;
%     semilogy(SNR_dB, BER_ELM(dop_idx, :), 'rs-', 'LineWidth', 2, 'DisplayName', 'ELM');
%     title(['BER vs SNR at Doppler Frequency = ', num2str(doppler_freqs(dop_idx)), ' Hz']);
%     xlabel('SNR (dB)');
%     ylabel('Bit Error Rate (BER)');
%     legend;
%     grid on;
%     ylim([1e-4 1]);
%     xlim([0 35]);
% end

for dop_idx = 1:length(doppler_freqs)
    figure;  % Create a new figure in each iteration
    semilogy(SNR_dB, BER_MMSE(dop_idx, :), 'bo-', 'LineWidth', 2, 'DisplayName', 'MMSE');
    hold on;
    semilogy(SNR_dB, BER_ELM(dop_idx, :), 'rs-', 'LineWidth', 2, 'DisplayName', 'ELM');
    title(['BER vs SNR at Doppler Frequency = ', num2str(doppler_freqs(dop_idx)), ' Hz']);
    xlabel('SNR (dB)');
    ylabel('Bit Error Rate (BER)');
    legend;
    grid on;
    ylim([1e-4 1]);
    xlim([0 35]);
end
