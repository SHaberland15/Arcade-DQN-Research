function [mat_out] = conv_with_gauss(mat_in, conv_param)

p = 0.9995;  % percentage covering gauss distribution

gauss_radius = ceil(norminv(p, 0, conv_param));

gauss_kernel = normpdf((-gauss_radius:gauss_radius), 0, conv_param)';
gauss_kernel = gauss_kernel/sum(gauss_kernel);

n_cut = size(gauss_kernel,1) - 1;

size_1 = size(mat_in,1);
size_2 = size(mat_in,2);
size_3 = size(mat_in,3);

size_1_cut = size_1 - n_cut;

mat_out = NaN(size_1_cut, size_2, size_3);

for k = 1:size_3
    
    for j = 1:size_2
        
        conv_temp = conv(mat_in(:,j,k), gauss_kernel, 'full');
        
        mat_out(:,j,k) = conv_temp(n_cut+1:size_1,:);
        
    end
    
end

