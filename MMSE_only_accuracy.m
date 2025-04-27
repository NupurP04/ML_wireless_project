% plot_MMSE_only.m
% Computes and plots MMSE detection accuracy vs. number of pilot subcarriers

clc; clear; close all;

%% Parameters
num_subcarriers = 64;       % OFDM subcarriers
cp_len          = 16;       % Cyclic prefix length
NR = 2; NT = 2;             % 2×2 MIMO
sigma_noise     = 0.2;      % AWGN scaling

%% Generate one OFDM frame
X      = randi([0 1], NT, num_subcarriers)*2 - 1;    % NT×64 BPSK symbols
X_OFDM = ifft(X, num_subcarriers, 2); 
X_CP   = [X_OFDM(:, end-cp_len+1:end), X_OFDM];     % add CP

%% Pass through MIMO channel + noise
H    = (randn(NR,NT) + 1j*randn(NR,NT))/sqrt(2);    % Rayleigh
Nmat = sigma_noise*(randn(NR,size(X_CP,2)) + 1j*randn(NR,size(X_CP,2)));
Y_CP = H*X_CP + Nmat;

%% Remove CP and FFT to get frequency‐domain Rx
Y      = Y_CP(:, cp_len+1:end);
Y_OFDM = fft(Y, num_subcarriers, 2);

%% Sweep pilot counts
num_pilot_values = 0:10:150;
mmse_accuracy    = zeros(size(num_pilot_values));

for idx = 1:length(num_pilot_values)
    P = min(num_pilot_values(idx), num_subcarriers);
    if P > 0
        % LS channel estimate on first P subcarriers
        Yp    = Y_OFDM(:,1:P);     % NR×P
        Xp    = X(:,1:P);          % NT×P
        H_est = Yp * pinv(Xp);     % NR×NT

        % Estimate noise variance from the time‐domain noise
        sigma2 = mean(abs(Nmat(:)).^2);

        % MMSE equalizer: G = (H'H + σ2 I)^(-1) H'
        G = (H_est'*H_est + sigma2*eye(NT)) \ H_est';

        % Detect all subcarriers
        X_hat = G * Y_OFDM;                         % NT×64
        X_pred = sign(real(X_hat));                 % decision

        % Accuracy over the full 64 subcarriers
        mmse_accuracy(idx) = mean(X_pred(:)==X(:))*100;
    else
        mmse_accuracy(idx) = 0;
    end
end

%% Plot MMSE accuracy vs. pilots
figure;
plot(num_pilot_values, mmse_accuracy, '-o', 'LineWidth', 2);
xlabel('Number of Pilot Subcarriers');
ylabel('MMSE Detection Accuracy (%)');
title('MMSE Accuracy vs. Number of Pilots');
grid on;
