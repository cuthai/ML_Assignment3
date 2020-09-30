Usage
	python main.py -dn <str> [-rs <int>] [-kt <str>] [-k <int>] [-s <float>] [-e <float>]

Args:
	-dn <str>
	Required, specifies the name of the data. Please use:
		glass
        segmentation
		vote
		abalone (note: this + -kt is very time consuming)
		forest-fires
		machine

	-rs <int>
	Optional, specifies the random_state of the data for splitting. Defaults to 1

	-kt
	Optional, specifies to run a specialized model of edited or condensed. Please use:
	    edited
	    condensed

    -k
	Optional, specifies a K to use for predicting

	-s
	Optional, specifies a sigma to use for predicting

	-e
	Optional, specifes an epsilon to use for predicting
