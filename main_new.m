    close all, clear all, clc

    x = 0.1:1/22:1;
    d = ((1 + 0.6*sin(2*pi*x/0.7)) + (0.3*sin(2*pi*x)))/2;

    % Mokymo zingsnis
    eta = 0.01;

    w11_1 = rand(8, 1);
    w12_1 = rand(8, 1);
    b1_1 = rand(8, 1);

    b2_1 = rand(1);

    w11_2 = rand(8, 1);
    w12_2 = rand(8, 1);
    b1_2 = rand(1);

    w2_2 = rand(1);

    for epoch = 1:200000 % epochu skaicius

        for i = 1:length(x)
            % Pirmas (pasleptasis) sluoksnis 
            y1_1 = tanh(w11_1 * x(i) + b1_1);
            y2_1 = tanh(w12_1 * x(i) + b2_1);

            % Isejimo sluoksnis
            y1_2 = w11_2' * y1_1 + w12_2' * y2_1 + b1_2;

            % Klaida
            e = d(i) - y1_2;

            % Skaicuojamas klaidos gradienatas
            delta1_2 = e;

            % Atsinaujiname svorius
            % Isejimo sluoksnis
            w11_2 = w11_2 + eta * delta1_2 * y1_1;
            w12_2 = w12_2 + eta * delta1_2 * y2_1;
            b1_2 = b1_2 + eta * delta1_2;

            % Pasleptasis sluoksnis
            for j = 1:8
                delta1_1(j) = (1 - tanh(y1_1(j))^2) * delta1_2 * w11_2(j);
                w11_1(j) = w11_1(j) + eta * delta1_1(j) * x(i);
                b1_1(j) = b1_1(j) + eta * delta1_1(j);
            end
        end
    end

    % inicializuojami masyvai laiko tinklo isejimu ir norimu isejimu
    output_network = zeros(size(x));
    output_desired = zeros(size(x));

    for i = 1:length(x)
        % Pirmas (pasleptasis) sluoksnis 
        y1_1 = tanh(w11_1 * x(i) + b1_1);
        y2_1 = tanh(w12_1 * x(i) + b2_1);

        % Isejimo sluoksnis
        y1_2 = w11_2' * y1_1 + w12_2' * y2_1 + b1_2;

        % Laikomas tinklo isejimas ir norimas isejimas
        output_network(i) = y1_2;
        output_desired(i) = d(i);
    end

    % Plot the network's output and the desired output
    figure;
    plot(x, output_network, 'b', x, output_desired, 'r');
    legend('Tinklo isejimas', 'Norimas isejimo rezultatas');
    xlabel('Iejimo duomenys (x)');
    ylabel('Isejimas');
    title('Neuroninio tinklo rezultatas lyginant su norimas rezultatu');
