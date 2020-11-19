function I=render_model_red(F,V,image_size,view_p,crop_box,camera_angle)
% Example use:
% 
% view_p=[1 1 1];
% image_size = 128;
% filename = 'depth_image.bmp';
% crop_box = 0;
% camera_angle = 7;
% render_model(F,V,image_size,view_p,crop_box,box_size,filename);


% rend = 'opengl';

if crop_box
    bb = 2.5 ;              % we confine the mesh model into a cube with side length 2*bb
else
    bb = max(abs(V(:)));    % we define the bounding box of the model as 2 times the furthest coordinate away from the center
end;

res = image_size/(2*bb);    % Define the resolution


%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define the view plane
r=10;                       % radius of the view sphere, should be set such that the camera position is outside the model range
n=view_p/norm(view_p);      % normal of the view plane
p=r*n;                      % camera position on the view sphere
a=n(1);                     % define the view plane parameters a, b, c, d
b=n(2);                     % ax + by + cz + d = 0
c=n(3);       
d=-dot(n,p);
%%%%%%%%%%%%%%%%%%%%%%%%%%




D_VP=abs(V*[a b c]'+d)/sqrt(a^2+b^2+c^2);  % calculate the distance of the vertices to the view-plane
D_VP_max=max(D_VP);     % find the maximum distance to the view-plane
DP=D_VP_max-D_VP;       % subtract to get the distance from the furthest point

DP=res*DP;              % make the depth values so that they are scaled with the same amount of 'x' and 'y' depth image coordinates 
DP=DP+1;                % make sure that '0' depth value does not occur (avoid merging with the background) 


% DP will serve as the colormap while rendering the mesh

figure('visible','off');
H=trisurf(F,V(:,1),V(:,2),V(:,3),DP);
set(H,'Edgealpha',0);               % set face edges invisible
set(H,'Facecolor','interp');        % interpolate the triangles using the vertex values
                                    % (bilinear interpolation)      
set(H,'CDataMapping','direct');
daspect([1 1 1])
colormap(gray(256))
axis off;


view(view_p);
set(gca,'CameraViewAngle',camera_angle);     % make sure that the camera angle is constant for all views
% set(gca,'CameraUpVector',camera_up_vector);
axis([-bb bb -bb bb -bb bb]);
set(gcf, 'color', 'red');
set(gcf, 'InvertHardCopy', 'off');
G=get(gca);
G_OP=G.OuterPosition;
set(gca,'Position',G_OP);
set(gcf,'PaperPosition', [0 0 2*bb 2*bb]);

% res = 2.5333*image_size;
res = 1.333*image_size;
% str=['print -dpng -' rend ' -r' num2str(res) ' ' filename];
I= print('-RGBImage', strcat('-r',num2str(res)));


% str=['print -dpng -' rend ' -r0 ' filename];

% eval(str);
close(gcf)





