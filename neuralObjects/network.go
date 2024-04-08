package neuralObjects

import "fmt"

type Network struct {
	inputLayer     NeuralLayer
	internalLayers []NeuralLayer
	outputLayer    NeuralLayer
}

func NewNetwork(inputLayerNodeCount int, interLayersNodeCounts []int, outputLayerNodeCount int) Network {
	network := Network{}
	network.inputLayer = NewNeuralLayer(inputLayerNodeCount, 0)

	prevNodeCount := inputLayerNodeCount

	for _, internalLayerNodeCount := range interLayersNodeCounts {
		network.internalLayers = append(network.internalLayers, NewNeuralLayer(internalLayerNodeCount, prevNodeCount))
		prevNodeCount = internalLayerNodeCount
	}

	network.outputLayer = NewNeuralLayer(outputLayerNodeCount, prevNodeCount)
	return network
}

func (network *Network) Update(inputs []float64) error {
	if len(inputs) != len(network.inputLayer.activation) {
		return fmt.Errorf("the number of inputs is incorrect for the network. txpect %d received %d", len(network.inputLayer.activation), len(inputs))
	}

	copy(network.inputLayer.activation, inputs)

	for i := 0; i < len(network.internalLayers); i++ {
		if i == 0 {
			network.internalLayers[i].Update(network.inputLayer)
		} else {
			network.internalLayers[i].Update(network.internalLayers[i-1])
		}

	}

	network.outputLayer.Update(network.internalLayers[len(network.internalLayers)-1])
	return nil
}

func (network *Network) GetOuputLayer() *NeuralLayer {
	return &network.outputLayer
}

func (network *Network) GetInputLayer() *NeuralLayer {
	return &network.inputLayer
}

func (network *Network) GetInternalLayer(i int) *NeuralLayer {
	return &network.internalLayers[i]
}

func (net *Network) PrintAllLayerActivations() {
	fmt.Println("The activations of the layers are")
	fmt.Println(net.inputLayer.activation)
	for _, l := range net.internalLayers {
		fmt.Println(l.activation)
	}
	fmt.Println(net.outputLayer.activation)
}
