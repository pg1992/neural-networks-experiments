clear all;
close all;
clc;

n_data = 500;
n_input = 2;
n_hid = 3;
n_class = 1;

w.input_to_hid = randn(n_hid, n_input);
w.hid_to_class = randn(n_class, n_hid);

x = randn(2, n_data);
t = (x(1, :) > 0 & x(2, :) > 0) | (x(1, :) < 0 & x(2, :) < 0);

y = simple_net(x, w);

subplot(121);
hold on;
plot(x(1, t), x(2, t), 'xk');
plot(x(1, ~t), x(2, ~t), 'or');
subplot(122);
hold on;
plot(x(1, y > .5), x(2, y > .5), 'xk');
plot(x(1, ~(y > .5)), x(2, ~(y > .5)), 'or');

epochs = 100000;
er = zeros(epochs, 1);
for i = 1:epochs
    w = train(x, t, w, .01);
    er(i) = cost(t, x, w);

    if mod(i, epochs / 10) == 0
        printf('%.3f%% of training completed.\n', i / epochs * 100);
    end
end
figure;
plot(er);

y = simple_net(x, w);
figure;
subplot(121);
hold on;
plot(x(1, t), x(2, t), 'xk');
plot(x(1, ~t), x(2, ~t), 'or');
subplot(122);
hold on;
plot(x(1, y > .5), x(2, y > .5), 'xk');
plot(x(1, ~(y > .5)), x(2, ~(y > .5)), 'or');
