package utils

import (
	"fmt"

	"golang.org/x/exp/constraints"
)

func MultiplyMatrixToVector[N constraints.Float | constraints.Integer](mat [][]N, vec []N) ([]N, error) {
	res := make([]N, len(mat))

	for i := 0; i < len(mat); i++ {
		if len(mat[i]) != len(vec) {
			return nil, fmt.Errorf("the matrices are not of the correct size. found %d elements at row %d but expectd %d", len(mat[i]), i, len(vec))
		}
		for j := 0; j < len(vec); j++ {
			res[i] += mat[i][j] * vec[j]
		}
	}

	return res, nil
}

func AddVectors[N constraints.Float | constraints.Integer](vec1 []N, vec2 []N) ([]N, error) {
	if len(vec1) != len(vec2) {
		return nil, fmt.Errorf("the vectors are different lengths %d, %d", len(vec1), len(vec2))
	}

	res := make([]N, len(vec1))
	for i := 0; i < len(vec1); i++ {
		res[i] = vec1[i] + vec2[i]
	}
	return res, nil
}
