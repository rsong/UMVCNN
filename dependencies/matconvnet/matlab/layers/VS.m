classdef VS< dagnn.Layer
    % @author: Ran Song
    properties
        vstride = 20;%standard setup is 20
        vfchannel=4096;
        hyper = 2;
        
    end
    
    methods
        
        function self = VS(varargin)
            self.load(varargin) ;
            self.vstride = self.vstride;
            self.vfchannel= self.vfchannel;
        end
        
        function outputs = forward(self, inputs, params )
            % -------------------------------------------------------------------------
            [sz1, sz2, sz3, sz4] = size(inputs{1});
            if mod(sz4,self.vstride)~=0 ,
                error('All shapes should have same number of views.');
            end
            
            if(sz1*sz2~=1)
                error ('The first 2 dimentions of the tensor should be 1x1.')
            end
         
            [coor,tri] = icosahedron2sphere(2);
            
      
            dist=sz3*sz4;
            inputfeats=reshape(inputs{1},[sz3 self.vstride sz4/self.vstride]);
            inputvind=reshape(inputs{2},[self.vstride sz4/self.vstride]);
            sa_tensor=zeros(self.vstride,sz4/self.vstride);
            for k=1:sz4/self.vstride
                for i=1:self.vstride
                    diffe=0;
                    [geodist,~,~]=perform_fast_marching_mesh(coor,tri,inputvind(i,k));
                    for j=1:self.vstride
                        if i~=j
                            difference=norm(inputfeats(:,i,k)-inputfeats(:,j,k))./(1+self.hyper*geodist(inputvind(j,k)));
                            diffe=diffe+difference;
                        end
                    end
                    
                    sa_tensor(i,k)=diffe;
                end
    
            end
            
            outputs{1} = reshape(sa_tensor/dist,[sz1 sz2 self.vstride sz4/self.vstride]);
        end
        
        
        % -------------------------------------------------------------------------
        
        function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
            % -------------------------------------------------------------------------
            [coor,tri] = icosahedron2sphere(2);
            
            [sz1, sz2, sz3, sz4] = size(inputs{1});
            inputfeats=reshape(inputs{1},[sz3 self.vstride sz4/self.vstride]);
            inputvind=reshape(inputs{2},[self.vstride sz4/self.vstride]);
        
            dist=sz3*sz4;

            numshape=sz4/self.vstride;
            
            if isa(derOutputs{1},'gpuArray')
                a=gpuArray(zeros(sz3,self.vstride,numshape));
            else
                a=zeros(sz3,self.vstride,numshape);
            end
          
            y=reshape(derOutputs{1},[self.vstride numshape]);
            
            
            for i=1:numshape
                for j=1:self.vstride
                    dydxj=zeros(sz3,1);
                    [geodist,~,~]=perform_fast_marching_mesh(coor,tri,inputvind(j,i));
                    for k=1:self.vstride
                        if j~=k
                        
                            dydxjk=(inputfeats(:,j,i)-inputfeats(:,k,i))/norm(inputfeats(:,j,i)-inputfeats(:,k,i)+eps);
                            dydxjk=dydxjk./(1+self.hyper*geodist(inputvind(k,i)));
                            dydxj=dydxj+dydxjk;
                        end
                    end
                    a(:,j,i)=y(j,i)*dydxj;
                end
            end
            
            derInputs{1}=reshape(a/dist,[sz1 sz2 sz3 sz4]);
            derInputs{2}=[];
            derParams={};
            
        end
    end
    
end


