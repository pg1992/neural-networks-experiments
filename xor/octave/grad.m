function dw = grad(x, t, w)

G = w.input_to_hid;
H = w.hid_to_class;

[n_class, n_hid] = size(H);
[n_dim, n_data] = size(x);

dw.input_to_hid = zeros(size(H));
dw.hid_to_class = zeros(size(G));

hid = sigmoid(G * x);
y = sigmoid(H * hid);

dout = (t - y) .* y .* (1 - y);

dw.hid_to_class =  -sum(repmat(dout, [n_hid 1]) .* hid, 2)';

dw_t = repmat(H', [1 n_data]) .* hid .* (1 - hid) .* repmat(dout, [n_hid 1]);
dw.input_to_hid = -dw_t * x';
