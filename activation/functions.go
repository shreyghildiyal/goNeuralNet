package activation

import (
	"log"
	"math"
)

type FunctionType int

const (
	ReLU FunctionType = iota
	FastSigmoid
)

func (a FunctionType) toString() string {
	switch a {
	case ReLU:
		return "ReLU"
	case FastSigmoid:
		return "FastSigmoid"
	default:
		return "Unknown"
	}
}

func Activate(funcType FunctionType, input float64) float64 {

	switch funcType {
	case FastSigmoid:
		return fastSigmoidActivation(input)
	default:
		log.Fatalf("Activation function %s not supported", funcType.toString())
		return 0
	}
}

func Derivative(funcType FunctionType, input, output float64) float64 {
	switch funcType {
	case FastSigmoid:
		return fastSigmoidDerivative(output)
	default:
		log.Fatalf("Derivative of function %s not supported", funcType.toString())
		return 0
	}
}

func fastSigmoidActivation(x float64) float64 {
	return x / (1 + math.Abs(x))
}

func fastSigmoidDerivative(y float64) float64 {
	oneMinusAbsY := 1.0 - math.Abs(y)
	return oneMinusAbsY * oneMinusAbsY
}
