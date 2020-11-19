function bv_view_p = bestview(filename)
% filename: the filename of a testing mesh
% bv_view_p: the xyz coordinates of the best viewpoint 
% Copyright (C) Ran Song

[vertex,face] = read_off(filename);

% Compute the saliency map
[p,t,knng,sa]= cnn_output_view_single(vertex,face);

% Compute the overall saliency from each view
V_P=subdivide_octa(3);
num_views=length(V_P);
mesh_dist=zeros(num_views,1);

for v=1:num_views
    % Only visible vertices have contributions to the overall saliency
    visibility_vs = mark_visible_vertices(p,t,V_P(v,:));
    visibility_v =visibility_vs(knng);

    total_mesh_saliency = sum(visibility_v.*sa);
    mesh_dist(v)=total_mesh_saliency;
end

% Get the rank of the viewpoints
[~, bv_geo_index]=max(mesh_dist);
bv_geo_index=bv_geo_index(1);
bv_view_p=V_P(bv_geo_index,:);

% [rankY,rankI]=sort(mesh_dist,'descend');
face=face';
vertex=vertex';

% Visualise the mesh from the best viewpoints
figure;
trimesh(face,vertex(:,1),vertex(:,2),vertex(:,3), 'FaceColor', 'c', 'EdgeColor', 'none', ...
    'AmbientStrength', 0.2, 'DiffuseStrength', 0.6, 'SpecularStrength', 0.1, 'FaceLighting', 'flat');
set(gcf, 'Color', 'w', 'Renderer', 'OpenGL');
set(gca, 'Projection', 'perspective');
axis equal ;axis off;view(bv_view_p);camlight('HEADLIGHT');

end

