package neuralObjects

import (
	"fmt"
	"math/rand"

	"github.com/shreyghildiyal/goNeuralNet/activation"
)

type NeuralLayer struct {
	weights            [][]float64
	biases             []float64
	activationFunction activation.FunctionType
	lastInputs         []float64
	lastOutputs        []float64
}

func (nl *NeuralLayer) Update(prevLayer NeuralLayer) {

}

func NewNeuralLayer(nodeCount int, prevLayerNodeCount int, activationFunction activation.FunctionType) NeuralLayer {
	layer := NeuralLayer{}
	layer.weights = make([][]float64, nodeCount)
	layer.biases = make([]float64, nodeCount)
	layer.lastInputs = make([]float64, prevLayerNodeCount)
	layer.lastOutputs = make([]float64, nodeCount)
	layer.activationFunction = activationFunction
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
