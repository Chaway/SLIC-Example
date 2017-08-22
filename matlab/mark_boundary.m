function [img,edge] = mark_boundary(origin,labels,color_rgb)

[row,col] = size(labels);
edge = zeros(row,col);
img = origin;

for i = 2:row - 1
    for j = 2:col -1
        if (labels(i-1:i+1,j-1:j+1) == labels(i,j))
            edge(i,j) = 1;
        else
            img(i,j,:) = color_rgb;
        end
    end
end

end



