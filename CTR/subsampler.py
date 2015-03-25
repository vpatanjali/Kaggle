for i in range(2,10):
	subsample_pct = 10*i
	infile = '../../Data/CTR/train.csv'
	outfile = '../../Data/CTR/train_%s_pct.csv' %(subsample_pct)

	#------------------------#

	import random

	inf = open(infile)
	outf = open(outfile,mode = 'w')

	header = inf.readline()
	outf.write(header)

	for line in inf:
		date = line.split(',')[2][4:6]
		if int(date) < 30 and random.randint(0,99) >= subsample_pct:
			continue
		outf.write(line)
	
	inf.close()
	outf.close()
	
