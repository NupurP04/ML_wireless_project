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
num_pilots = 2;                  % Minimal pilots
num_symbols = N_fft - num_pilots - 1; % Number of data symbols
numBits = 50 * Nt * num_symbols; % Total bits
fs = 5e6;                        % Sampling frequency (Hz)
fd = 50;                         % Doppler shift (Hz)
Pt = 1;                          % Transmit power
time_stamps = numBits / (Nt * num_symbols); % Number of OFDM symbols
SNR_dB = 0:5:25;                 % SNR range (dB)
L = 16;                          % Number of hidden neurons for ELM
R = 2;                           % Transmission rate: 2 bps/Hz for 2x2 MIMO with BPSK

%% MIMO Channel Object
rayleighChan = comm.MIMOChannel( ...
    'FadingDistribution', 'Rayleigh', ...
    'SampleRate', fs, ...
    'MaximumDopplerShift', fd, ...
    'NumTransmitAntennas', Nt, ...
    'NumReceiveAntennas', Nr, ...
    'SpatialCorrelationSpecification', 'None');

%% Preallocate Arrays
BER_MMSE = zeros(length(SNR_dB), 1);
BER_ELM = zeros(length(SNR_dB), 1);
spectral_efficiency_MMSE = zeros(length(SNR_dB), 1);
spectral_efficiency_ELM = zeros(length(SNR_dB), 1);

%% ELM Parameters
alpha = 0.01;                    % Regularization parameter
activation = @(x) 1./(1+exp(-x)); % Sigmoid activation function
W = unifrnd(-1,1,L,2*Nr);        % Random weights (L x 2*Nr for real/imag parts)
B = unifrnd(-1,1,L,1);           % Random biases
X_P = [1 1; 1 -1; -1 1; -1 -1].'; % All possible BPSK MIMO symbol vectors

%% Generate Random Bits
bits = randi([0 1], numBits, 1);
symbols = 2 * bits - 1;          % BPSK mapping: 0 -> -1, 1 -> 1

%% SNR Loop
for snr_idx = 1:length(SNR_dB)
    snr_linear = 10^(SNR_dB(snr_idx)/10);
    noise_variance = Pt / snr_linear;
    noise_std = sqrt(noise_variance/2);

    % Transmitter Side
    ofdm_symbols = zeros(N_fft, Nt, time_stamps);
    ofdm_no_pilot_symbols = pagetranspose(reshape(symbols, Nt, num_symbols, []));
    
    % Insert Orthogonal Pilots
    ofdm_symbols(1,1,:) = 1;    ofdm_symbols(1,2,:) = 1;    % Pilot 1
    ofdm_symbols(2,1,:) = 1;    ofdm_symbols(2,2,:) = -1;   % Pilot 2
    k = (N_fft/2) - num_pilots;
    for i = 1:time_stamps
        ofdm_symbols(num_pilots+1:N_fft/2,:,i) = ofdm_no_pilot_symbols(1:k,:,i);
        ofdm_symbols(N_fft/2+2:end,:,i) = ofdm_no_pilot_symbols(k+1:end,:,i);
    end

    % IFFT and Add CP
    ofdm_symbols_time = ifft(ofdm_symbols, N_fft);
    ofdm_symbols_cp = cat(1, ofdm_symbols_time(end-cp_len+1:end,:,:), ofdm_symbols_time);
    txFrame = sqrt(Pt) * ofdm_symbols_cp;

    detected_MMSE = zeros(size(ofdm_no_pilot_symbols));
    detected_ELM = zeros(size(ofdm_no_pilot_symbols));

    for i = 1:time_stamps
        % Channel
        H_actual = (randn(Nr,Nt)+1j*randn(Nr,Nt))/sqrt(2);
        rxFrame = zeros(size(txFrame(:,:,i)));
        for n = 1:N_fft+cp_len
            rxFrame(n,:) = rayleighChan(txFrame(n,:,i));
        end

        % Add Noise
        noise = noise_std*(randn(size(rxFrame)) + 1j*randn(size(rxFrame)));
        rxFrame = rxFrame + noise;

        % Receiver Side
        rxFrame_no_cp = rxFrame(cp_len+1:end, :);
        rxFrame_freq = fft(rxFrame_no_cp, N_fft);

        % Channel Estimation with Pilots
        Y_pilot = rxFrame_freq(1:num_pilots,:).';
        X_pilot = squeeze(ofdm_symbols(1:num_pilots,:,i)).';
        H_est = Y_pilot * pinv(X_pilot);

        % Data Detection
        data_idx = [num_pilots+1:N_fft/2, N_fft/2+2:N_fft];
        rx_data = rxFrame_freq(data_idx,:);

        for h = 1:length(data_idx)
            Y = rx_data(h,:).';

            % MMSE Detection
            G_MMSE = (H_est'*H_est + noise_variance*eye(Nt)) \ (H_est');
            X_hat_MMSE = G_MMSE * Y;
            detected_MMSE(h,:,i) = sign(real(X_hat_MMSE));

            % ELM Detection
            Y_P = zeros(4,Nr);
            for k_p = 1:4
                Y_P(k_p,:) = (H_est) * X_P(:,k_p);
            end
            M_P = zeros(4,L);
            for k_p = 1:4
                input_k = [real(Y_P(k_p,:)), imag(Y_P(k_p,:))];
                M_P(k_p,:) = activation(W * input_k' + B).';
            end
            beta_k = zeros(4,Nt,L);
            for k_p = 1:4
                norm_Mk = sum(M_P(k_p,:).^2) + alpha;
                beta_k(k_p,:,:) = X_P(:,k_p) * M_P(k_p,:) / norm_Mk;
            end
            input_n = [real(Y.'), imag(Y.')];
            M_n = activation(W * input_n' + B);
            distances = zeros(4,1);
            for k_p = 1:4
                distances(k_p) = norm(Y - Y_P(k_p,:).');
            end
            [~,k_min] = min(distances);
            X_hat_ELM = squeeze(beta_k(k_min,:,:)) * M_n;
            detected_ELM(h,:,i) = sign(real(X_hat_ELM));
        end
    end

    % BER Calculation
    transmitted_data = ofdm_no_pilot_symbols;
    errors_MMSE = sum(transmitted_data(:) ~= detected_MMSE(:));
    errors_ELM = sum(transmitted_data(:) ~= detected_ELM(:));
    BER_MMSE(snr_idx) = errors_MMSE/numBits;
    BER_ELM(snr_idx) = errors_ELM/numBits;

    % Spectral Efficiency
    spectral_efficiency_MMSE(snr_idx) = R * (1 - BER_MMSE(snr_idx));
    spectral_efficiency_ELM(snr_idx) = R * (1 - BER_ELM(snr_idx));
end

%% Plot Spectral Efficiency
figure;
plot(SNR_dB, spectral_efficiency_MMSE, 'bo-', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'DisplayName', 'MMSE');
hold on;
plot(SNR_dB, spectral_efficiency_ELM, 'rs-', 'LineWidth', 2, 'MarkerFaceColor', 'r', 'DisplayName', 'ELM (L=16)');
xlabel('SNR (dB)');
ylabel('Spectral Efficiency (bps/Hz)');
title('Spectral Efficiency vs SNR for MMSE and ELM (L=16)');
grid on;
legend('Location', 'southeast');
ylim([0 2]);
xlim([0 25]);
