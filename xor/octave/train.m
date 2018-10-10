function new_w = train(x, t, w, lr)

gr = grad(x, t, w);
new_w.input_to_hid = w.input_to_hid - lr * gr.input_to_hid;
new_w.hid_to_class = w.hid_to_class - lr * gr.hid_to_class;
