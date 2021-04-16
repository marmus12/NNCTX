

output_dir = '/media/emre/Data/DATA/redandblack/';
filename = 'redandblack_vox10_1450';
filepath = ['/media/emre/Data/DATA/redandblack/redandblack/Ply/' filename '.ply'];
pc =pcread(filepath);

write_arr_to_file(pc.Location,[output_dir filename '.dat']);

GT = pc.Location;
Loc = GT-min(GT)+32;
lrGT = unique(floor(Loc/2),'rows');

write_arr_to_file(lrGT,[output_dir filename '_9.dat']);