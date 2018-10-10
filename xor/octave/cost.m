function E = cost(t, x, w)

y = simple_net(x, w);
E = .5 * sum((t(:) - y(:)) .^ 2);
