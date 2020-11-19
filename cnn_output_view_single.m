
function [p,t,knng,sa]= cnn_output_view_single(v,f)
%%
% Compute the saliency map sa of a mesh represented by vertex matrix v and
% face matrix f;
% p and t are vertices and faces of the simplified mesh;
% knng contains the mapping relation between p and v.

% Copyright (C) Ran Song
% All rights reserved.
%%
modelPath = '.\pretrained\net-deployed-bestv.mat' ;

net=load(modelPath') ;
net = dagnn.DagNN.loadobj(net) ;

outputSize = 224;
net.mode = 'test' ;
nViews=20;

% Designate the layers that we to hijack in a forwardprop
predVar_sa = net.getVarIndex('xViewSa') ;
predVar_im = net.getVarIndex('input') ;
predVar = net.getVarIndex('obprob') ;

nChannels = size(net.params(1).value,3);

averageImage = net.meta.normalization.averageImage;
if numel(averageImage)==nChannels
    averageImage = reshape(averageImage, [1 1 nChannels]);
end

fig = figure('Visible','off');

mesh.V=v;
mesh.F=f;

xn1=max(mesh.V(1,:));
xn2=min(mesh.V(1,:));
yn1=max(mesh.V(2,:));
yn2=min(mesh.V(2,:));
zn1=max(mesh.V(3,:));
zn2=min(mesh.V(3,:));

mesh.V(1,:)=mesh.V(1,:)-0.5*(xn1+xn2);
mesh.V(2,:)=mesh.V(2,:)-0.5*(yn1+yn2);
mesh.V(3,:)=mesh.V(3,:)-0.5*(zn1+zn2);

% Mesh simplication for efficiency
if length(mesh.F)>2000
    [p,t] = perform_mesh_simplification(mesh.V',mesh.F',2000);
    knng=knnsearch(p,mesh.V');
else
    p=mesh.V';
    t=mesh.F';
    knng=1:length(mesh.V);
end

% Vertex density
W=density(t,p);

mesh1.V=p';
mesh1.F=t';
vv=mesh.V';
ff=mesh.F';

%% Generate the 20 candidate views
[ims,viewpoints20,rank_v] = get_optviews(vv,ff);

imsvsas=zeros(length(p),nViews);

imsvsa=zeros(length(mesh.V),nViews);
[a,b,c]=size(ims{1});
im=zeros(a,b,c,numel(ims),'single');

for j=1:numel(ims)
    im(:,:,:,j) = ims{j};
end
im=single(im);

im_ = bsxfun(@minus, im,averageImage);
viewinds=rank_v(1:nViews);

inputs = {'input', im_, 'viewind', viewinds};

%% Infer the net and get the outputs of the designated layers
net.vars(predVar_im).precious=1;

net.vars(predVar_sa).precious=1;
net.vars(predVar).precious=1;

net.eval(inputs) ;

vweights=gather(net.vars(predVar_sa).value) ;
vweights=vweights(:);

scores = gather(net.vars(predVar).value) ;
[~,ind_class]=max(scores,[],3);

%% Backprop the net to get the 2D saliency maps
dzdy=zeros(1,1,30,1);

dzdy(1,1,ind_class,1)=1;
deout={'obprob',dzdy};

net.eval(inputs,deout) ;
aaa = gather(net.vars(predVar_im).der) ;

%% Generate multiview saliency maps
msa= imsvsa;

for ii=1:nViews
    % Plot the mesh with the 20 viewpoints
    plotMesh(mesh1,viewpoints20(ii,:));
    
    [az,el] = view;
    cam_angle=camva;
    
    %% 2D-to-3D saliency mapping
    
    %Crop the 2D image
    [crop,r1,r2,c1,c2]=autocrop(ims{ii});

    aa=size(crop,1);
    bb=size(crop,2);
    longside=max(aa,bb);
    shortside=min(aa,bb);
    
    bbb=aaa(:,:,:,ii);

    ccc=max(abs(bbb),[],3);

    ddd=(ccc-min(ccc(:)))./(max(ccc(:))-min(ccc(:)));
    
    % Deliver the projection
    T=viewmtx(az,el,cam_angle);
    x4d=[p';ones(1,length(p))];
    x2d = T*x4d;

    xxa=max(x2d(1,:));
    xxi=min(x2d(1,:));
    yya=max(x2d(2,:));
    yyi=min(x2d(2,:));
    
    xx=xxa-xxi;
    yy=yya-yyi;
    
    longx=max(xx,yy);
    shortx=min(xx,yy);
    
    scale1=longside/longx;
    scale2=shortside/shortx;
    
    sscale=0.5*(scale1+scale2);
    x1=x2d(1,:)*sscale;
    y1=x2d(2,:)*sscale;
    
    x2=x1+0.5*outputSize;
    y2=y1-0.5*outputSize;
    
    % Get the ROI
    cropsa=ddd(r1:r2,c1:c2);
    
    y2=-y2;
    x2=x2-min(x2);% col;
    y2=y2-min(y2);% row;

    vpx=viewpoints20(ii,1);
    vpy=viewpoints20(ii,2);
    vpz=viewpoints20(ii,3);
    
    % Get vertex visibility and smooth it a bit to make sure that the
    % visibles vertices are reliably detected
    visibility_v = mark_visible_vertices(p,t,[vpx,vpy,vpz]);
    visibility_v=perform_mesh_smoothing(t,p,visibility_v);
    visibility_v=perform_mesh_smoothing(t,p,visibility_v);
    
    %Find 2D-3D correspondence for the mapping
    [impointsx,impointsy]=meshgrid(1:bb,1:aa);
    impoints=[impointsx(:) impointsy(:)];
    
    visible=find(visibility_v~=0);
    invisible=find(visibility_v==0);
   
    vx2=x2(visible);
    vy2=y2(visible);
    x2ddd=[vx2(:) vy2(:)];
    ind_cor = knnsearch(impoints,x2ddd);
    
    for jj=1:length(visible)
        
        row=impoints(ind_cor(jj),2);
        col=impoints(ind_cor(jj),1);
        % Visible vertices are assigned with the corresponding 2D saliency
        imsvsas(visible(jj),ii)=exp(1-single(W(visible(jj))))/(exp(1-cropsa(row,col)));
        
    end
    % Invisible vertices are assigned with the mean saliency of the visible vertices
    imsvsas(invisible,ii)=mean(imsvsas(visible,ii));
     
    imsvsas(:,ii)= perform_mesh_smoothing(t,p,imsvsas(:,ii));
    % Map the saliency from the simplified mesh back to the original mesh
    imsvsa(:,ii)=imsvsas(knng,ii);

    %% Saliency weighted by the outputs of the SP layer
    msa(:,ii)=imsvsa(:,ii).*vweights(ii);
end

%% Multiview saliency aggregation and normalisation
sa=sum(msa,2);
close(fig);
close all;
sa(isnan(sa))=min(sa);
sa=(sa-min(sa))/(max(sa)-min(sa));
end

