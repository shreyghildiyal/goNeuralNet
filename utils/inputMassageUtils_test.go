package utils_test

import (
	"testing"

	"github.com/shreyghildiyal/goNeuralNet/utils"
)

func TestNormalizeLabels(t *testing.T) {

	labels := []int{1, 2, 3}
	numLabels := 10
	res := utils.NormalizeLabels(labels, numLabels)

	for i, val := range labels {
		if len(res[i]) != numLabels {
			t.Errorf("Length or response for index %d is wrong. expected %d, found %d", i, numLabels, len(res[i]))
			t.Fail()
		}

		for j, v := range res[i] {

			if j == val && v != 1 {
				t.Errorf("Expected %dth entry in output to be 1 but was %f", j, v)
			}
			if j != val && v != 0 {
				t.Errorf("Expected %dth entry in output to be 0 but was %f", j, v)
			}
		}
	}
}
