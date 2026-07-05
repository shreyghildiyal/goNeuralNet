package utils

func MaxIndex(slice []float64) int {
	bestIndex := 0

	for i, val := range slice {
		if val > slice[bestIndex] {
			bestIndex = i
		}
	}
	return bestIndex
}
