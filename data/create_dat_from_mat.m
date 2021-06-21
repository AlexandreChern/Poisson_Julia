N_list = [2^3,2^4,2^5,2^6,2^7,2^8];

for n = N_list
    input_A = sprintf('A_%d.mat',n);
    input_b = sprintf('b_%d.mat',n);
    output_A = sprintf('A_%d.dat',n);
    output_b = sprintf('b_%d.dat',n);
%     load fullfile(input_A);
%     load fullfile(input_b);
%     importdata(input_A);
%     importdata(input_b);
%     disp(A);
%     disp(b);
    
    PetscBinaryWrite(output_A,importdata(input_A));
    PetscBinaryWrite(output_b,importdata(input_b));
end
