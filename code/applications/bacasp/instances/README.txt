Each instance is designated by the name "R_N_i", where "N" is the number of vessels 
and "i" identifies the number of the instance. For all instances, a time 
horizon of 60 time periods is considered.

The information displayed in each file "R_N_i" is as follows:

-> First  line "HORA_CHEGADA": vector of vessel arrival times
-> Second line: Empty
-> Third  line "QUANTIDADE": vector corresponding to the load on board of each vessel
-> Forth  line: Empty
-> Fifth  line "TAXA": vector of crane processing rates
-> Sixth  line: Empty
-> Seven  line "INICIO": vector of lower berth positions in which each crane can operate  
-> Eight  line: Empty
-> Ninth  line "FIM": vector of upper berth positions in which each crane can operate  
-> Tenth  line: Empty
-> Eleventh line "COMPRIMENTO": vector of lengths of the vessels.


*All the 100 instances presented are instances with homogeneous cranes. Heterogeneous    
 instances can be obtained by changing the vector "TAXA".
From the paper: The heterogeneous cranes instances were obtained from the ho-
mogeneous instances by increasing the processing rate of cranes
3 and 4 from 263.644 to 319.001. The processing rate of the re-
maining cranes 1, 2, 5, 6, and 7 are kept unchanged and equal to
263.644 units per hour.