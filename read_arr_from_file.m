
function [arr,sz1,sz2] = read_arr_from_file(fpath)

    fid = fopen(fpath,'rb');
    
    sz1 = fread(fid,1,'int32');
    sz2 = fread(fid,1,'int32');
    farr = fread(fid,inf,'int32');
    fclose(fid);
    arr = reshape(farr,[sz1,sz2]);
    arr=arr';