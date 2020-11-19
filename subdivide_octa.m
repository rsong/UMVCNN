
function [new_vertices,new_faces]=subdivide_octa(num_levels)



view_ps=[ 1 0 0 ; -1 0 0 ; 0 1 0 ; 0 -1 0 ; 0 0 1 ; 0 0 -1];
faces=[...
      5 2 3
      5 3 1
      5 1 4
      5 4 2
      6 2 3
      6 3 1
      6 1 4
      6 4 2
   ];

   
if num_levels == 0;
    new_vertices=view_ps;
    new_faces=faces;
    return;
end;
 
 
 for l=1:num_levels;
     
      new_vertices=view_ps;  
      num_v=length(new_vertices);
 
      new_faces=[];
      num_f=length(faces);
 
 for f=1:num_f;
     
     v1_id=faces(f,1);
     v2_id=faces(f,2);
     v3_id=faces(f,3);
     
     v1=view_ps(v1_id,:);
     v2=view_ps(v2_id,:);
     v3=view_ps(v3_id,:);
     
     va=(v1+v2)/2;
     va=va/norm(va);
     vb=(v1+v3)/2;
     vb=vb/norm(vb);
     vc=(v2+v3)/2;
     vc=vc/norm(vc);
     
     [a,loc]= ismember(va,new_vertices,'rows'); 
     if a;
         va_id=loc;
     else
         new_vertices=[new_vertices;va];
         num_v=num_v+1;
         va_id=num_v;
     end;
     
     [b,loc]= ismember(vb,new_vertices,'rows');
     if b;
         vb_id=loc;
     else
         new_vertices=[new_vertices;vb];
         num_v=num_v+1;
         vb_id=num_v;
     end;
     
     [c,loc]= ismember(vc,new_vertices,'rows');    
     if c;
         vc_id=loc;
     else
         new_vertices=[new_vertices;vc];
         num_v=num_v+1;
         vc_id=num_v;
     end;
     
     
     new_faces=[new_faces; v1_id va_id vb_id; v2_id va_id vc_id; v3_id vc_id vb_id; va_id vb_id vc_id];
     
          
 end;
 
 view_ps=new_vertices;
 faces=new_faces;
 
 end;
 
 

     