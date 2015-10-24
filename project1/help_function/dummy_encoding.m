function X_out = dummy_encoding(X_in)
dum_values = unique(X_in);
X_out = zeros(size(X_in,2),length(dum_values));
for i = 1:length(dum_values)
  v = dum_values(i);
  index = find(X_in == v);
  X_out_sub = zeros(size(X_in,2),1);
  X_out_sub(index) = 1;
  X_out(:,i) = X_out_sub;
end

