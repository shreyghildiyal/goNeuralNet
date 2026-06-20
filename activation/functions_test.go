package activation_test

import (
	"math"
	"testing"

	"github.com/shreyghildiyal/goNeuralNet/activation"
)

var sigmoidActivationTestData = []struct {
	input    float64
	expected float64
}{
	{input: 0.0, expected: 0.0},
	{input: 1.0, expected: 0.5},
	{input: -1.0, expected: -0.5},
}

var sigmoidDerivativeTestData = []struct {
	input    float64
	output   float64
	expected float64
}{
	{input: math.NaN(), output: 0.0, expected: 1.0},
	{input: math.Inf(1), output: 0.5, expected: 0.25},
	{input: math.Inf(-1), output: -0.5, expected: 0.25},
}

func TestActivateFastSigmoid(t *testing.T) {

	for _, testCase := range sigmoidActivationTestData {
		actual := activation.Activate(activation.FastSigmoid, testCase.input)
		if actual != testCase.expected {
			t.Errorf("The derivative is not correct. Expected: %f Found: %f", testCase.expected, actual)
		}
	}

}

func TestFastSigmoidDerivative(t *testing.T) {
	for _, testCase := range sigmoidDerivativeTestData {
		actual := activation.Derivative(activation.FastSigmoid, testCase.input, testCase.output)
		if actual != testCase.expected {
			t.Errorf("The derivative is not correct. Expected: %f Found: %f", testCase.expected, actual)
		}
	}
}
