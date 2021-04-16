function BS = read_bs(file_path)

fileID = fopen(file_path, 'r');
BS = fread(fileID, inf, 'ubit1');
fclose(fileID);

