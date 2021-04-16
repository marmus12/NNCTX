
function vec = read_vec_from_file(fpath)

    fid = fopen(fpath,'rb');
    
    sz1 = fread(fid,1,'int32');

    vec = fread(fid,inf,'int32');
    fclose(fid);

