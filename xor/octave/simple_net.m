function y = simple_net(x, w)

h = sigmoid(w.input_to_hid * x);
y = sigmoid(w.hid_to_class * h);
