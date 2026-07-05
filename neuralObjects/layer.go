package neuralObjects

import (
	"math/rand"

	"github.com/shreyghildiyal/goNeuralNet/activation"
)

type NeuralLayer struct {
	weights                [][]float64
	biases                 []float64
	activationFunction     activation.FunctionType
	lastUnactivatedOutputs []float64
	lastOutputs            []float64

	// Cumulative batch-level buckets (Gradients)
	cumulativeWeightErrors [][]float64 // Shape: [nodes][prevNodes] (Must match weights!)
	cumulativeBiasErrors   []float64   // Shape: [nodes]     (Must match biases!)

	// current errors
	currentWeightErrors [][]float64 // Shape: [nodes][prevNodes] (Must match weights!)
	currentBiasErrors   []float64   // Shape: [nodes]     (Must match biases!)

}

func (nl *NeuralLayer) ApplyAndResetGradients(batchSize float64, rate float64) {
	// panic("unimplemented")
	for i := 0; i < len(nl.weights); i++ {
		for j := 0; j < len(nl.weights[i]); j++ {
			nl.weights[i][j] += rate * nl.cumulativeWeightErrors[i][j] / batchSize
			nl.cumulativeWeightErrors[i][j] = 0
		}
		nl.biases[i] += rate * nl.cumulativeBiasErrors[i] / batchSize
		nl.cumulativeBiasErrors[i] = 0
	}
}

func (nl *NeuralLayer) ForwardUpdate(prevLayer *NeuralLayer) {

	// copy(nl.lastUnactivatedOutputs, prevLayer.lastOutputs)

	for i := 0; i < len(nl.lastOutputs); i++ {
		val := 0.0
		for j := 0; j < len(nl.weights[i]); j++ {
			val += nl.weights[i][j] * prevLayer.lastOutputs[j]
		}
		val += nl.biases[i]
		nl.lastUnactivatedOutputs[i] = val
		nl.lastOutputs[i] = activation.Activate(nl.activationFunction, val)
	}
}

func (nl *NeuralLayer) BackPropagate(incomingErrors []float64, prevLayer *NeuralLayer) []float64 {

	numNodes := len(nl.weights)
	prevLayerErrors := make([]float64, len(prevLayer.lastOutputs))

	nodeDeltas := make([]float64, numNodes)

	for i := 0; i < numNodes; i++ {
		z := nl.lastUnactivatedOutputs[i]
		a := nl.lastOutputs[i]
		nodeDeltas[i] = incomingErrors[i] * activation.Derivative(nl.activationFunction, z, a)
	}

	for i := 0; i < numNodes; i++ {
		delta := nodeDeltas[i]
		nl.currentBiasErrors[i] = delta
		nl.cumulativeBiasErrors[i] += delta

		for j := 0; j < len(prevLayer.lastOutputs); j++ {
			weightGrad := delta * prevLayer.lastOutputs[j]
			nl.currentWeightErrors[i][j] = weightGrad
			nl.cumulativeWeightErrors[i][j] += weightGrad

			prevLayerErrors[j] += delta * nl.weights[i][j]
		}

	}

	return prevLayerErrors
}

func NewNeuralLayer(nodeCount int, prevLayerNodeCount int, activationFunction activation.FunctionType) NeuralLayer {
	layer := NeuralLayer{}
	layer.weights = make([][]float64, nodeCount)
	layer.biases = make([]float64, nodeCount)
	layer.lastUnactivatedOutputs = make([]float64, nodeCount)
	layer.lastOutputs = make([]float64, nodeCount)

	layer.cumulativeWeightErrors = make([][]float64, nodeCount)
	layer.currentWeightErrors = make([][]float64, nodeCount)
	layer.cumulativeBiasErrors = make([]float64, nodeCount)
	layer.currentBiasErrors = make([]float64, nodeCount)

	layer.activationFunction = activationFunction
	for i := 0; i < nodeCount; i++ {
		layer.weights[i] = make([]float64, prevLayerNodeCount)
		layer.cumulativeWeightErrors[i] = make([]float64, prevLayerNodeCount)
		layer.currentWeightErrors[i] = make([]float64, prevLayerNodeCount)
		for j := 0; j < prevLayerNodeCount; j++ {
			layer.weights[i][j] = rand.Float64()
		}
		layer.biases[i] = rand.Float64()
		// layer.nodes = append(layer.nodes, NewNode(prevLayerNodeCount))
	}
	// fmt.Println("New Layer created with weights", layer.weights)
	return layer
}
