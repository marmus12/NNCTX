function write_arr_to_file(arr,fpath)

    arr=arr';
   %% mkdir(fpath);
    [w,h] = size(arr);
    fid = fopen(fpath,'wb');
    
    fwrite(fid,w,'int32');
    fwrite(fid,h,'int32');
    fwrite(fid,arr(:),'int32');

    fclose(fid);


end

