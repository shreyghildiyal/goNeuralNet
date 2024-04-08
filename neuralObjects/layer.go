package neuralObjects

import (
	"fmt"
	"math/rand"
)

type NeuralLayer struct {
	weights            [][]float64
	biases             []float64
	activation         []float64
	prevLayerNodeCount int
}

func (nl *NeuralLayer) Update(prevLayer NeuralLayer) {
	var sum float64
	for i := 0; i < len(nl.weights); i++ {
		sum = 0
		for j := 0; j < nl.prevLayerNodeCount; j++ {
			sum += nl.weights[i][j] * prevLayer.activation[j]
		}
		nl.activation[i] = sum + nl.biases[i]
	}
	// fmt.Println("After update we have activation", nl.activation)
}

func NewNeuralLayer(nodeCount int, prevLayerNodeCount int) NeuralLayer {
	layer := NeuralLayer{}
	layer.prevLayerNodeCount = prevLayerNodeCount
	layer.weights = make([][]float64, nodeCount)
	layer.biases = make([]float64, nodeCount)
	layer.activation = make([]float64, nodeCount)
	for i := 0; i < nodeCount; i++ {
		layer.weights[i] = make([]float64, prevLayerNodeCount)
		for j := 0; j < prevLayerNodeCount; j++ {
			layer.weights[i][j] = rand.Float64()
		}
		layer.biases[i] = rand.Float64()
		// layer.nodes = append(layer.nodes, NewNode(prevLayerNodeCount))
	}
	fmt.Println("New Layer created with weights", layer.weights)
	return layer
}

func (layer *NeuralLayer) GetActivations() []float64 {
	return layer.activation
}
