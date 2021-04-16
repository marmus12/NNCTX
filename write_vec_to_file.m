function write_vec_to_file(vec,fpath)

   %% mkdir(fpath);
    lvec = length(vec);
    fid = fopen(fpath,'wb');
    
    fwrite(fid,lvec,'int32');

    fwrite(fid,vec,'int32');

    fclose(fid);


end

