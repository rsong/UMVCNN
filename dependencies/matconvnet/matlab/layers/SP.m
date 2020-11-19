classdef SP < dagnn.Layer
    properties
        vstride = 20;
        vfchannel=4096;
    end
    
    methods
        function self =SP(varargin)
            self.load(varargin) ;
            self.vstride = self.vstride;
            self.vfchannel= self.vfchannel;
        end
        function outputs = forward(self, inputs, params)
            
            gpuMode = isa(inputs{1}, 'gpuArray');
            [sz1, sz2, sz3a, sz4] = size(inputs{1});
            if mod(sz4,self.vstride)~=0 ,
                error('All shapes should have same number of views.');
            end
            
            if(sz1*sz2~=1)
                error ('The first 2 dimentions of the tensor should be 1x1.')
            end
            [~, ~, sz3b, ~] = size(inputs{2});%1 x 1 x sz3b x sz4/self.vstride
            
            if(sz3b~=self.vstride)
                error ('The view-saliency tensor should have n channels where n is the number of views.')
            end
            
            tensor1 = reshape(inputs{1},[sz1 sz2 sz3a self.vstride sz4/self.vstride]);
            tensor2 = reshape(inputs{2},[sz1 sz2 sz3b 1 sz4/self.vstride]);
            
            if gpuMode
                y = gpuArray(zeros([1, 1, sz3a, sz4/sz3b], 'single'));
            else
                y = zeros([1, 1, sz3a, sz4/sz3b], 'single');
            end
            
            for i = 1:sz4/sz3b,
                
                xa = reshape(tensor1(:,:,:,:,i), [sz3a,sz3b]);
                xb = reshape(tensor2(:,:,:,:,i), [sz3b,1]);
                
                y(1,1,:, i) = reshape(xa*xb, [1 sz3a]);
            end
            
            outputs{1} = y;
        end
        
        function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
            gpuMode = isa(inputs{1}, 'gpuArray');
            [sz1, sz2, sz3a, sz4] = size(inputs{1});
            [~, ~, sz3b, ~] = size(inputs{2});%1 x 1 x sz3b x sz4/self.vstride; sz3a=4096; sz3b=self.vstride;
            if gpuMode
                y1 = gpuArray(zeros([sz1, sz2, sz3a, sz4], 'single'));
                y2 = gpuArray(zeros([1, 1, sz3b, sz4/self.vstride], 'single'));
            else
                y1 = zeros([sz1, sz2, sz3a, sz4], 'single');
                y2 = zeros([1, 1, sz3b, sz4/self.vstride], 'single');
            end
            
            y1=reshape(y1, [sz1, sz2, sz3a, sz3b, sz4/self.vstride]);
            
            dzdyr=reshape(dzdy, [sz3a 1 sz4/self.vstride]);
            xa = reshape(inputs{1}, [sz3a self.vstride sz4/self.vstride]);
            xb= reshape(inputs{2},[self.vstride 1 sz4/self.vstride]);
            for i=1:sz4/self.vstride
                x1=xa(:,:,i);
                x2=xb(:,:,i);
                dzdyv=dzdyr(:,:,i);
                db = reshape(x1'*dzdyv, [1, 1, sz3b]);
                da = reshape(dzdyv*x2', [1, 1, sz3a,sz3b]);
                
                y1(:,:,:,:,i)=da;
                y2(:,:,:,i)=db;
            end
            y1=reshape(y1,[1,1,sz3a,sz4]);
            derInputs{1}=y1;
            derInputs{2}=y2;
            derParams = {};
            
        end
           
    end
end
