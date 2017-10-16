function mat=readMat(filename, type)

if (sum(strcmp(type, {'txt', 'text'})))
    % import from space separated file and skip the first line.
    mat = importdata(filename, ' ', 1);
    mat = mat.data;
%     mat = dlmread(filename, ' ', 1, 0);
else
    % read binary file
    file = fopen(filename);
    r = fread(file,1,'int');
    d = fread(file,1,'int');
    mat=fread(file,r*d, type);
    mat = reshape(mat,d,r); % Row major , column major difference
    mat = mat';
    fclose(file);
end

end