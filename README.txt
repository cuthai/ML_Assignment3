Usage
	python main.py -dn <str> [-p] [-t] [-pt <float>]

Args:
	-dn <str>
	Required, specifies the name of the data. Please use:
		breast-cancer
		car
        	segmentation
		abalone (note: this + -t is very time consuming)
		forest-fires
		machine

	-rs <int>
	Optional, specifies the random_state of the data for splitting. Defaults to 1

	-p
	Optional, specifies pruning on classification datasets (works on breast-cancer, car, segmentation)

    	-t
	Optional, specifies tuning on regression datasets (works on abalone, forest-fires, machine)

	-pt
	Optional, specifies a percent_threshold to use as early stopping criteria on regression datasets (works on abalone, forest-fires, machine)
