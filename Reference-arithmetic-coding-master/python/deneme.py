# 
# Compression application using adaptive arithmetic coding
# 
# Usage: python adaptive-arithmetic-compress.py InputFile OutputFile
# Then use the corresponding adaptive-arithmetic-decompress.py application to recreate the original input file.
# Note that the application starts with a flat frequency table of 257 symbols (all set to a frequency of 1),
# and updates it after each byte encoded. The corresponding decompressor program also starts with a flat
# frequency table and updates it after each byte decoded. It is by design that the compressor and
# decompressor have synchronized states, so that the data can be decompressed properly.
# 
# Copyright (c) Project Nayuki
# 
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
# 

import contextlib, sys
import arithmeticcoding


# # Command line main application function.
# def main(args):
# 	# Handle command line arguments
# 	if len(args) != 2:
# 		sys.exit("Usage: python adaptive-arithmetic-compress.py InputFile OutputFile")
# 	inputfile, outputfile = args
# 	
# 	# Perform file compression
# 	with open(inputfile, "rb") as inp, \
# 			contextlib.closing(arithmeticcoding.BitOutputStream(open(outputfile, "wb"))) as bitout:
# 		compress(inp, bitout)


outputfile = 'outfile.dat'
bitout = contextlib.closing(arithmeticcoding.BitOutputStream(open(outputfile, "wb"))) 

initfreqs = arithmeticcoding.FlatFrequencyTable(257)
freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
enc = arithmeticcoding.ArithmeticEncoder(32, bitout)




symbol = 1
enc.write(freqs, symbol)
freqs.increment(symbol)
enc.write(freqs, 256)  # EOF
enc.finish()  # Flush remaining code bits


# # Main launcher
# if __name__ == "__main__":
# 	main(sys.argv[1 : ])
