package neuralObjects

import "github.com/shreyghildiyal/goNeuralNet/activation"

type Network struct {
	layers       []NeuralLayer
	learningRate float64
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

func (network *Network) Train(images [][]float64, labels [][]float64, epochs, batchSize int) {

	// one epoch runs through the entire dataset once.

	for i := 0; i < epochs; i++ {
		for j := 0; j < len(images); j += batchSize {
			network.runBatch(images, labels, j, batchSize)
		}
	}
	// one batch runs through upto a certain number of input values before updating the network values

}

func (network *Network) runBatch(images [][]float64, labels [][]float64, startIndex, batchsize int) {

	// create an array to hold the total errors we see
	// errArr := make([]float64, len(network.layers[len(network.layers)-1].weights))

	for i := 0; i < batchsize && startIndex+i < len(images); i++ {
		network.forwardPass(images[startIndex+i])
		network.backPropagate(labels[startIndex+i])
	}

	for i := 1; i < len(network.layers); i++ {
		network.layers[i].ApplyAndResetGradients(float64(batchsize), network.learningRate)
	}
}

func (network *Network) backPropagate(targetLabel []float64) {

	errSlice := make([]float64, len(targetLabel))

	layersCount := len(network.layers)

	for i := 0; i < len(targetLabel); i++ {
		errSlice[i] = targetLabel[i] - network.layers[layersCount-1].lastOutputs[i]
	}

	for i := len(network.layers) - 1; i > 0; i-- {
		errSlice = network.layers[i].BackPropagate(errSlice, network.layers[i-1])
	}

	// panic("unimplemeted")
}

func (network *Network) forwardPass(input []float64) {

	// set the values as the output of the input layer
	for i := range network.layers[0].lastOutputs {
		network.layers[0].lastOutputs[i] = input[i]
	}

	for i := 1; i < len(network.layers); i++ {
		network.layers[i].ForwardUpdate(&network.layers[i-1])
	}
}
