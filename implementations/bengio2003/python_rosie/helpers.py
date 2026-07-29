def matmul(m1: list, m2: list):
	"""
	Parameters
	----------
	m1 : list
		m1 matrix, must be a row vector (1 dim list) or 2 dim matrix (list of lists)
	m2 : list
		m2 matrix, must be a column vector (1 dim list) or 2 dim matrix (list of lists)
	"""
	if not isinstance(m1[0], list): # assume row vector
		m1 = [m1]
	if not isinstance(m2[0], list): # assume column vector
		m2 = [[x] for x in m2]

	o = [] # m1.shape[1], m2.shape[0]

	for i in range(len(m1)):
		o.append([])

		for j in range(len(m2[0])):
			# extract cols from m2
			col = []
			for row in m2:
				col.append(row[j])

			assert len(col) == len(m1[i])

			# do the matrix multiplication
			sum = 0
			for xi, xj in zip(col, m1[i]):
				sum += xi * xj

			# store output
			o[i].append(sum)
		assert len(o[i]) == len(m2[0])
	assert len(o) == len(m1)
	return o


def transpose(m: list):
	"""
    Parameters
    ----------
    m : list[list]
        m matrix, must be a 1 or 2 dim matrix (list of lists)
    """
	if not isinstance(m[0], list): # assume row vector
		m = [m]
	return [list(row) for row in zip(*m)]