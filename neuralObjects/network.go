package neuralObjects

import "github.com/shreyghildiyal/goNeuralNet/activation"

type Network struct {
	// inputLayer     NeuralLayer
	layers []NeuralLayer
	// outputLayer    NeuralLayer
}

func NewNetwork(layerCounts []int) Network {
	network := Network{}
	// network.layers = NewNeuralLayer(inputLayerNodeCount, 0, activation.FastSigmoid)

	// prevNodeCount := inputLayerNodeCount
	prevNodeCount := 1

	for _, layerNodeCount := range layerCounts {
		network.layers = append(network.layers, NewNeuralLayer(layerNodeCount, prevNodeCount, activation.FastSigmoid))
		prevNodeCount = layerNodeCount
	}

	// network.outputLayer = NewNeuralLayer(outputLayerNodeCount, prevNodeCount, activation.FastSigmoid)
	return network
}

func (network *Network) GetOuputLayer() *NeuralLayer {
	return &network.layers[len(network.layers)-1]
}

func (network *Network) GetInputLayer() *NeuralLayer {
	return &network.layers[0]
}

func (network *Network) GetLayer(i int) *NeuralLayer {
	return &network.layers[i]
}
