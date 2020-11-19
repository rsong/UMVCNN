function I_new=read_red_bckg(II,A)
% II=imread(filename);
red_in=find((II(:,:,1)==255).*(II(:,:,2)==0).*(II(:,:,3)==0));
I_new=rgb2gray(II);
I_new(red_in)=-A;

