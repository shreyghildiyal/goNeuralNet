package utils

import "math"

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func RelU(x float64) float64 {
	if x <= 0 {
		return 0
	} else {
		return x
	}
}
