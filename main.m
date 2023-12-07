close all, clear all, clc

x = 0.1:1/22:1;
d = ((1 + 0.6*sin(2*pi*x/0.7)) + (0.3*sin(2*pi*x)))/2;

% mokymo zingsnis
eta = 0.01;

% svoriai ir bias pasleptajam sluoksniui
w_1A = rand(1);
w_1B = rand(1);
w_1C = rand(1);
w_1D = rand(1);

b_A = rand(1);
b_B = rand(1);
b_C = rand(1);
b_D = rand(1);

% svoriai ir bias isejimo sluoksniui
w_A2 = rand(1);
w_B2 = rand(1);
w_C2 = rand(1);
w_D2 = rand(1);

b_2 = rand(1);

% kiek kartu leisti mokyma
num_epochs = 500000;

% Mokymo ciklas
for epoch = 1:num_epochs
    for i = 1:length(x) % Dirbama su kiekvienu 
        % Aktyvavimo funkcija pirmame sluoksnyje
        y_A = tanh(w_1A * x(i) + b_A);
        y_B = tanh(w_1B * x(i) + b_B);
        y_C = tanh(w_1C * x(i) + b_C);
        y_D = tanh(w_1D * x(i) + b_D);
        
        % Pasverta suma isejiimo sluoksnyje
        y_2 = w_A2*y_A + w_B2*y_B + w_C2*y_C + w_D2*y_D + b_2;
        
        % Palyginti su norimu atsaku ir apskaiciuoti klaida
        e = d(i) - y_2;
        
        % Backpropagation
        % Atnaujinti rysiu svorius
        delta_2 = e; % Output layer gradient
        delta_A = (1 - y_A.^2) * (w_A2 * delta_2); % Hidden layer gradients
        delta_B = (1 - y_B.^2) * (w_B2 * delta_2);
        delta_C = (1 - y_C.^2) * (w_C2 * delta_2);
        delta_D = (1 - y_D.^2) * (w_D2 * delta_2);
        
        % Atnaujinti svorius isejimo sluoksnyje
        w_A2 = w_A2 + eta * delta_2 * y_A;
        w_B2 = w_B2 + eta * delta_2 * y_B;
        w_C2 = w_C2 + eta * delta_2 * y_C;
        w_D2 = w_D2 + eta * delta_2 * y_D;
        
        b_2 = b_2 + eta * delta_2;
        
        % Atnaujinti svorius pasleptame sluoksnyje
        w_1A = w_1A + eta * delta_A * x(i);
        w_1B = w_1B + eta * delta_B * x(i);
        w_1C = w_1C + eta * delta_C * x(i);
        w_1D = w_1D + eta * delta_D * x(i);
        
        b_A = b_A + eta * delta_A;
        b_B = b_B + eta * delta_B;
        b_C = b_C + eta * delta_C;
        b_D = b_D + eta * delta_D;
    end
end

% Testuojame
x_test = 0:1/222:1;
Y_end = zeros(size(x_test));

for i = 1:length(x_test)
    y_A = tanh(w_1A * x_test(i) + b_A);
    y_B = tanh(w_1B * x_test(i) + b_B);
    y_C = tanh(w_1C * x_test(i) + b_C);
    y_D = tanh(w_1D * x_test(i) + b_D);
    
    Y_end(i) = w_A2*y_A + w_B2*y_B + w_C2*y_C + w_D2*y_D + b_2;
end

% Plot the results
figure;
plot(x, d, 'g', x_test, Y_end, 'r');
xlabel('Iejimas');
ylabel('Isejimas');
legend('Tikimasis', 'Sugeneruotas tinklo');
title('Daugiasluoksnio perceptrono aproksimacija');
