function [ims,viewpoints20,rank_v] = get_optviews(vv,ff)
%% Render a mesh from 20 views subject to view area and silhouette length
% vv: vertice of the input mesh
% ff: faces of the input mesh
% ims: rendered 2D images
% viewpoint20: the 20 viewpoints
% rank_v: the goodness ranking of the 20 viewpoints

% Copyright (C) Ran Song
% All rights reserved.
%%
% Check and normalise the mesh
[v,f]=removedupnodes(vv,ff);
[f,~,~]=unique(sort(f')','rows');
xn1=max(v(:,1));
xn2=min(v(:,1));
yn1=max(v(:,2));
yn2=min(v(:,2));
zn1=max(v(:,3));
zn2=min(v(:,3));

v(:,1)=v(:,1)-0.5*(xn1+xn2);
v(:,2)=v(:,2)-0.5*(yn1+yn2);
v(:,3)=v(:,3)-0.5*(zn1+zn2);

bbox=max(abs(v(:)));
v=v./bbox;

mesh.V=v';
mesh.F=f';

image_size = 224;
crop_box = 0;
camera_angle=8;

[viewpoint,~] = icosahedron2sphere(2);
num_view=length(viewpoint);

%% Compute the view area and the silhouette length of each view
view_area=zeros(num_view,1);
sil_length=zeros(num_view,1);

for i=1:num_view
    %  view area
    I=render_model_red(f,v,image_size,viewpoint(i,:),crop_box,camera_angle);
    I_view=read_red_bckg(I,1);
    I_view=I_view>0;
    view_area(i)=sum(I_view(:));
    
    % silhouette length
    B = bwboundaries(I_view,8,'noholes');
    b_lengths=zeros(length(B),1);
    for b=1:length(B)
        S=B{b};
        b_lengths(b)=size(S,1);
    end
    [~,S_I]=max(b_lengths);
    S=B{S_I(1)};
    S_L = sum ( sqrt ( ( S(2:end,1)-S(1:end-1,1) ).^2 + ( S(2:end,2)-S(1:end-1,2) ).^2 ) );
    sil_length(i)=S_L;
end

%% Get the rank subject to both view area and silhouette length
[~,ind_view_area]=sort(view_area);
[~,ind_sil_length]=sort(sil_length);

rview_area=zeros(num_view,1);
rsil_length=zeros(num_view,1);

for r=1:num_view
    rview_area(r)=find(ind_view_area==r);
    rsil_length(r)=find(ind_sil_length==r);
end
r_all=rview_area+rsil_length;
[~,rank_v]=sort(r_all,'descend');
viewpoints20=viewpoint(rank_v(1:20),:);

%% Do the rendering
ims = cell(1,20);
for i=1:20
    renderMesh(mesh,'solid',viewpoints20(i,:));
    ims{i} = print('-RGBImage', '-r100');  %in case of an error,you have an old matlab version: comment this line and uncomment the following 2 ones
    ims{i} = resize_im(ims{i}, image_size, 0.1, 0.3);
end

end

%% Image resizing
function im = resize_im(im,outputSize,minMargin,maxArea)

max_len = outputSize * (1-minMargin);
max_area = outputSize^2 * maxArea;

nCh = size(im,3);
mask = ~im2bw(im,1-1e-10);
mask = imfill(mask,'holes');
% blank image (all white) is outputed if not object is observed
if isempty(find(mask, 1))
    im = uint8(255*ones(outputSize,outputSize,nCh));
    return;
end
[ys,xs] = ind2sub(size(mask),find(mask));
y_min = min(ys); y_max = max(ys); h = y_max - y_min + 1;
x_min = min(xs); x_max = max(xs); w = x_max - x_min + 1;
scale = min(max_len/max(h,w), sqrt(max_area/sum(mask(:))));
patch = imresize(im(y_min:y_max,x_min:x_max,:),scale);
[h,w,~] = size(patch);
im = uint8(255*ones(outputSize,outputSize,nCh));
loc_start = floor((outputSize-[h w])/2);
im(loc_start(1)+(0:h-1),loc_start(2)+(0:w-1),:) = patch;

end