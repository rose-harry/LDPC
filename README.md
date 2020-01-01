# LDPC Decoder.
Implementation of an LDPC-decoder using Loopy Belief Propagation for Binary Symmetric Channel.

The original message is located in the first 252 bits of the decoded signal. Recover the original English message by reading off the first 248 bits of the 252-bit message and treating them as a sequence of 31 ASCII symbols.


https://www.ics.uci.edu/~welling/teaching/ICS279/LPCD.pdf
http://kom.aau.dk/~tlj/mpa_lp_decoding_ldpc.pdf
